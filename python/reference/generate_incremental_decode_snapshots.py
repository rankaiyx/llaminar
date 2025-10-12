#!/usr/bin/env python3
"""
Generate incremental decode reference snapshots with true KV cache.

This module generates token-by-token snapshots for incremental decode,
using HuggingFace's native past_key_values (KV cache) mechanism.

Unlike the original decode snapshot generator which replays the full sequence,
this implementation does TRUE incremental decode:
- Each token is processed individually
- KV cache is maintained and grown incrementally
- Snapshots are saved per-token for direct comparison with Llaminar

@author David Sanftenberg
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
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


def incremental_decode_with_prefill(
    model_path: str,
    prefill_tokens: List[int],
    num_decode_tokens: int,
    verbose: bool = False
) -> Tuple[Dict[int, Dict[str, np.ndarray]], List[int]]:
    """
    Run incremental decode with prefill+decode split (matches Llaminar's approach).
    Run incremental decode with prefill+decode phases, matching Llaminar's behavior.
    
    This performs:
    1. PREFILL: Process prefill_tokens to build KV cache (snapshots NOT saved)
    2. DECODE: Generate num_decode_tokens, capturing snapshots for each
    
    Args:
        model_path: Path to GGUF model file
        prefill_tokens: Tokens to process during prefill (e.g., [1,2,3,4,5])
        num_decode_tokens: Number of tokens to generate during decode (e.g., 3)
        verbose: Whether to print progress
        
    Returns:
        Tuple of:
        - Dictionary mapping decode_token_index -> {stage_name: numpy_array}
          (token_0 = first decode token, token_1 = second decode token, etc.)
        - List of greedy-sampled decode tokens
        
    Example:
        # Prefill with [1,2,3,4,5], then generate 3 tokens
        snapshots, sampled_tokens = incremental_decode_with_prefill(
            "models/qwen.gguf",
            prefill_tokens=[1,2,3,4,5],
            num_decode_tokens=3
        )
        # snapshots[0] = stages for first generated token
        # snapshots[1] = stages for second generated token
        # snapshots[2] = stages for third generated token
        # sampled_tokens = [next_token_1, next_token_2, next_token_3]
    """
    if verbose:
        print(f"Starting incremental decode with prefill+decode...")
        print(f"  Model: {model_path}")
        print(f"  Prefill tokens: {prefill_tokens} ({len(prefill_tokens)} tokens)")
        print(f"  Decode tokens: {num_decode_tokens}")
    
    # Load model
    capturer = PipelineStageCapture(model_path, verbose=verbose)
    capturer.load_model()
    
    # Storage
    decode_snapshots = {}
    sampled_tokens = []
    past_key_values = None
    
    # PHASE 1: PREFILL (process all tokens as a batch, don't save snapshots)
    if verbose:
        print(f"\n{'='*60}")
        print(f"PHASE 1: PREFILL ({len(prefill_tokens)} tokens)")
        print(f"{'='*60}")
        print(f"  Processing prefill batch: {prefill_tokens}")
    
    # Process all prefill tokens in a single batch (matching Llaminar's prefill behavior)
    prefill_snapshots = capturer.capture_stages(prefill_tokens, past_key_values=None)
    past_key_values = capturer.past_key_values if hasattr(capturer, 'past_key_values') else None
    
    if verbose:
        cache_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        print(f"  ✓ Prefill complete, KV cache length: {cache_len}")
    
    # Save prefill KV cache for debugging
    if past_key_values is not None and len(past_key_values) > 0:
        prefill_cache_dir = workspace_dir / "pytorch_snapshots_mapped" / "prefill_cache"
        prefill_cache_dir.mkdir(parents=True, exist_ok=True)
        
        sample_k_shape = None
        sample_v_shape = None
        
        for layer_idx, (k_cache, v_cache) in enumerate(past_key_values):
            # k_cache shape: [batch, n_kv_heads, seq_len, head_dim]
            # v_cache shape: [batch, n_kv_heads, seq_len, head_dim]
            k_np = k_cache.detach().cpu().numpy()
            v_np = v_cache.detach().cpu().numpy()
            
            if layer_idx == 0:
                sample_k_shape = k_np.shape
                sample_v_shape = v_np.shape
            
            np.save(prefill_cache_dir / f"K_CACHE_layer{layer_idx}.npy", k_np)
            np.save(prefill_cache_dir / f"V_CACHE_layer{layer_idx}.npy", v_np)
        
        if verbose:
            print(f"  ✓ Saved prefill KV cache to {prefill_cache_dir}")
            print(f"    {len(past_key_values)} layers saved")
            if sample_k_shape is not None:
                print(f"    K cache shape: {sample_k_shape} (per layer)")
                print(f"    V cache shape: {sample_v_shape} (per layer)")
    
    # PHASE 2: DECODE (generate tokens and save snapshots)
    if verbose:
        print(f"\n{'='*60}")
        print(f"PHASE 2: DECODE ({num_decode_tokens} tokens)")
        print(f"{'='*60}")
    
    # Get initial logits from the last prefill output (don't re-process!)
    logits = prefill_snapshots.get('LM_HEAD') if prefill_snapshots is not None else None
    
    if verbose and logits is not None:
        print(f"  Prefill logits shape: {logits.shape}")
        print(f"  Sampling from position: {logits.shape[1] - 1} (last token)")
    
    for decode_idx in range(num_decode_tokens):
        
        if logits is not None:
            # Greedy sampling - use last position for batch, first position for incremental
            if logits.shape[1] > 1:
                # Batch prefill: use last sequence position
                next_token = int(logits[0, -1, :].argmax())
                if verbose and decode_idx == 0:
                    print(f"  Prefill logits[-1] top 5: {logits[0, -1, :].argsort()[-5:][::-1]}")
                    print(f"  Selected token: {next_token}")
            else:
                # Incremental decode: single position
                next_token = int(logits[0, 0, :].argmax())
            sampled_tokens.append(next_token)
            
            if verbose:
                print(f"\n  Decode step {decode_idx + 1}/{num_decode_tokens}")
                print(f"    Generated token: {next_token}")
                cache_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
                print(f"    KV cache length: {cache_len}")
            
            # Capture snapshots for this decode token
            token_snapshots = capturer.capture_stages([next_token], past_key_values=past_key_values)
            past_key_values = capturer.past_key_values if hasattr(capturer, 'past_key_values') else None
            
            # Save snapshots for this decode step
            decode_snapshots[decode_idx] = token_snapshots
            
            # Get logits for next iteration
            logits = token_snapshots.get('LM_HEAD')
            
            if verbose:
                print(f"    ✓ Captured {len(token_snapshots)} stages")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"✓ Decode complete!")
        print(f"  Generated tokens: {sampled_tokens}")
        print(f"  Snapshots saved: {len(decode_snapshots)}")
    
    # Cleanup
    capturer.cleanup()
    
    return decode_snapshots, sampled_tokens


def incremental_decode_with_cache(
    model_path: str,
    token_sequence: List[int],
    verbose: bool = False
) -> Tuple[Dict[int, Dict[str, np.ndarray]], List[int]]:
    """
    Run incremental decode token-by-token with KV cache, capturing all stages.
    
    This performs TRUE incremental decode:
    1. Process each token individually
    2. Use past_key_values (KV cache) from previous token
    3. Capture all pipeline stages for each token
    4. Save KV cache state after each token
    5. Track greedy-sampled tokens for sequence validation
    
    Args:
        model_path: Path to GGUF model file
        token_sequence: Complete token sequence to process (e.g., [1,2,3,4,5,6])
        verbose: Whether to print progress
        
    Returns:
        Tuple of:
        - Dictionary mapping token_index -> {stage_name: numpy_array}
        - List of greedy-sampled tokens (for validation)
        
    Example:
        # Process sequence [1,2,3,4,5,6]
        snapshots, sampled_tokens = incremental_decode_with_cache(
            "models/qwen.gguf",
            token_sequence=[1,2,3,4,5,6]
        )
        # snapshots[0] = stages for token 1
        # snapshots[1] = stages for token 2
        # ...
        # sampled_tokens = [7890, 1234, 5678, ...] (what model would generate)
    """
    if verbose:
        print(f"Starting incremental decode with KV cache...")
        print(f"  Model: {model_path}")
        print(f"  Token sequence: {token_sequence}")
        print(f"  Total tokens: {len(token_sequence)}")
    
    # Load model
    capturer = PipelineStageCapture(model_path, verbose=verbose)
    capturer.load_model()
    
    # Storage for all token snapshots
    all_token_snapshots = {}
    
    # Storage for greedy-sampled tokens (for validation against Llaminar)
    sampled_tokens = []
    
    # KV cache state (past_key_values from HuggingFace)
    past_key_values = None
    
    # Process each token incrementally
    for token_idx, token_id in enumerate(token_sequence):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Token {token_idx + 1}/{len(token_sequence)}: {token_id}")
            print(f"  KV cache: {'present' if past_key_values is not None else 'empty'}")
            if past_key_values is not None:
                # past_key_values is tuple of (key, value) per layer
                cache_len = past_key_values[0][0].shape[2]  # seq_len dimension
                print(f"  Cache length: {cache_len} tokens")
        
        # Capture pipeline stages using capturer (with KV cache)
        # This runs the model once and captures all intermediate stages via hooks
        # AND updates the KV cache for the next iteration
        token_snapshots = capturer.capture_stages([token_id], past_key_values=past_key_values)
        
        # DEBUG: Check captured tensor shapes
        if verbose and token_idx == 0:
            print(f"\n  [DEBUG] Captured tensor shapes:")
            for stage_name, tensor in list(token_snapshots.items())[:10]:
                print(f"    {stage_name}: {tensor.shape}")
        
        # Get updated KV cache from capturer for next token
        past_key_values = capturer.past_key_values if hasattr(capturer, 'past_key_values') else None
        
        # Extract greedy-sampled token for validation
        logits = token_snapshots.get('LM_HEAD')
        if logits is not None:
            # Greedy sampling: argmax of logits
            next_token = int(logits[0, 0, :].argmax())
            sampled_tokens.append(next_token)
            
            if verbose:
                next_token_logit = float(logits[0, 0, :].max())
                print(f"  Logits shape: {logits.shape}")
                print(f"  Greedy sampled token: {next_token} (logit={next_token_logit:.4f})")
        
        # Store snapshots for this token
        all_token_snapshots[token_idx] = token_snapshots
        
        if verbose:
            print(f"  ✓ Captured {len(token_snapshots)} stages")
            stage_names = list(token_snapshots.keys())[:5]
            if len(stage_names) < len(token_snapshots):
                stage_names.append(f"... (+{len(token_snapshots) - 5} more)")
            print(f"    Stages: {', '.join(stage_names)}")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"✓ Incremental decode complete!")
        print(f"  Total tokens processed: {len(all_token_snapshots)}")
        print(f"  Snapshots per token: {len(all_token_snapshots[0]) if all_token_snapshots else 0}")
        print(f"  Sampled tokens: {sampled_tokens}")
    
    # Cleanup
    capturer.cleanup()
    
    return all_token_snapshots, sampled_tokens


def save_prefill_logits(
    prefill_snapshots: Dict[str, np.ndarray],
    output_dir: Path,
    verbose: bool = False
) -> None:
    """
    Save prefill logits for debugging divergence issues.
    
    Args:
        prefill_snapshots: Dictionary of prefill stage snapshots
        output_dir: Output directory
        verbose: Print progress
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the LM_HEAD (logits) from prefill
    if 'LM_HEAD' in prefill_snapshots:
        logits = prefill_snapshots['LM_HEAD']
        
        # logits shape: [batch, seq_len, vocab_size]
        # We want the last position: [batch, vocab_size]
        if logits.shape[1] > 1:
            last_logits = logits[0, -1, :]  # Shape: [vocab_size]
        else:
            last_logits = logits[0, 0, :]
        
        logits_file = output_dir / "prefill_logits.npy"
        np.save(logits_file, last_logits)
        
        if verbose:
            print(f"\n✓ Saved prefill logits to {logits_file}")
            print(f"  Shape: {last_logits.shape}")
            print(f"  Top 5 tokens: {last_logits.argsort()[-5:][::-1]}")
    else:
        if verbose:
            print(f"\n⚠ Warning: No LM_HEAD in prefill snapshots")


def save_incremental_snapshots(
    snapshots: Dict[int, Dict[str, np.ndarray]],
    sampled_tokens: List[int],
    output_dir: Path,
    verbose: bool = False
) -> None:
    """
    Save incremental decode snapshots to disk.
    
    Directory structure:
        output_dir/
            sampled_tokens.json  # Greedy-sampled token sequence
            token_0/
                EMBEDDING.npy
                ATTENTION_OUTPUT_layer0.npy
                ...
            token_1/
                EMBEDDING.npy
                ...
            ...
    
    Args:
        snapshots: Dictionary from incremental_decode_with_cache()
        sampled_tokens: List of greedy-sampled tokens for validation
        output_dir: Base output directory
        verbose: Print progress
    """
    import json
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sampled tokens JSON
    tokens_file = output_dir / "sampled_tokens.json"
    with open(tokens_file, 'w') as f:
        json.dump({
            "sampled_tokens": sampled_tokens,
            "num_tokens": len(sampled_tokens),
            "description": "Greedy-sampled tokens from incremental decode"
        }, f, indent=2)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Saving snapshots to: {output_dir}")
        print(f"  Tokens: {sampled_tokens}")
        print(f"  Number of token snapshots: {len(snapshots)}")
        print(f"{'='*60}")
    
    # Save snapshots for each token
    for token_idx, stage_dict in sorted(snapshots.items()):
        token_dir = output_dir / f"token_{token_idx}"
        token_dir.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print(f"\nSaving token_{token_idx} ({len(stage_dict)} stages)...")
        
        for stage_name, stage_data in stage_dict.items():
            stage_path = token_dir / f"{stage_name}.npy"
            np.save(stage_path, stage_data)
            
            if verbose and len(stage_dict) <= 10:  # Only show details for small snapshots
                print(f"  {stage_name}: {stage_data.shape}")
    
    if verbose:
        print(f"\n✓ Saved {len(snapshots)} token snapshots to {output_dir}/")
        print(f"✓ Saved sampled tokens to {tokens_file}")


def save_model_weights(
    model_path: str,
    output_dir: Path,
    verbose: bool = False
) -> None:
    """
    Save attention layer weights from PyTorch model for verification.
    
    Saves Q, K, V, O projection weights for all layers to enable direct
    comparison with Llaminar's loaded weights. This eliminates uncertainty
    about weight loading, quantization, or sharding differences.
    
    Directory structure:
        output_dir/
            weights/
                layer0_Q_WEIGHT.npy  # [d_model, num_heads * head_dim]
                layer0_K_WEIGHT.npy  # [d_model, num_kv_heads * head_dim]
                layer0_V_WEIGHT.npy  # [d_model, num_kv_heads * head_dim]
                layer0_O_WEIGHT.npy  # [num_heads * head_dim, d_model]
                layer1_Q_WEIGHT.npy
                ...
    
    Args:
        model_path: Path to GGUF model file
        output_dir: Base output directory (weights/ will be created inside)
        verbose: Print progress
    """
    if verbose:
        print(f"\nSaving model weights from {model_path}")
    
    # Load model
    from reference import ModelRegistry
    model = ModelRegistry.create("qwen", model_path)
    model.load_model()
    hf_model = model.hf_model
    
    # Create weights directory
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== Save embedding table first ==========
    if verbose:
        print(f"  Extracting embedding table...")
    
    if hasattr(hf_model.model, 'embed_tokens'):
        # Get embedding weight: shape [vocab_size, d_model]
        embedding_weight = hf_model.model.embed_tokens.weight.detach().cpu().numpy()
        np.save(weights_dir / "token_embd.weight.npy", embedding_weight)
        
        if verbose:
            print(f"    Embedding shape: {embedding_weight.shape}")
            print(f"    Embedding[0,:5]: {embedding_weight[0,:5]}")
            print(f"    Embedding[1,:5]: {embedding_weight[1,:5]}")
            print(f"  ✓ Saved embedding table")
    else:
        print(f"  ⚠ Warning: Could not find embed_tokens in model")
    
    # ========== Save layer weights ==========
    if not hasattr(hf_model.model, 'layers'):
        raise RuntimeError("Cannot find transformer layers in model")
    
    layers = hf_model.model.layers
    num_layers = len(layers)
    
    if verbose:
        print(f"  Extracting weights from {num_layers} layers...")
    
    for layer_idx, layer in enumerate(layers):
        if not hasattr(layer, 'self_attn'):
            raise RuntimeError(f"Layer {layer_idx} missing self_attn")
        
        attn = layer.self_attn
        
        # Extract weight matrices (these are torch.nn.Linear layers)
        # Weight shape is typically [out_features, in_features] in PyTorch
        q_weight = attn.q_proj.weight.detach().cpu().numpy()  # [num_heads * head_dim, d_model]
        k_weight = attn.k_proj.weight.detach().cpu().numpy()  # [num_kv_heads * head_dim, d_model]
        v_weight = attn.v_proj.weight.detach().cpu().numpy()  # [num_kv_heads * head_dim, d_model]
        o_weight = attn.o_proj.weight.detach().cpu().numpy()  # [d_model, num_heads * head_dim]
        
        # Save to disk
        np.save(weights_dir / f"layer{layer_idx}_Q_WEIGHT.npy", q_weight)
        np.save(weights_dir / f"layer{layer_idx}_K_WEIGHT.npy", k_weight)
        np.save(weights_dir / f"layer{layer_idx}_V_WEIGHT.npy", v_weight)
        np.save(weights_dir / f"layer{layer_idx}_O_WEIGHT.npy", o_weight)
        
        if verbose and layer_idx == 0:
            print(f"    Layer 0 weight shapes:")
            print(f"      Q: {q_weight.shape}")
            print(f"      K: {k_weight.shape}")
            print(f"      V: {v_weight.shape}")
            print(f"      O: {o_weight.shape}")
            print(f"      Q[0,:10]: {q_weight[0,:10]}")
            print(f"      K[0,:10]: {k_weight[0,:10]}")
    
    if verbose:
        print(f"  ✓ Saved weights for {num_layers} layers to {weights_dir}/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate incremental decode snapshots with true KV cache"
    )
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to GGUF model file"
    )
    
    # Mode 1: Process all tokens (original behavior)
    parser.add_argument(
        "--tokens",
        default=None,
        help="Comma-separated token sequence to process all at once (e.g., '1,2,3,4,5,6')"
    )
    
    # Mode 2: Prefill + decode (new behavior, matches Llaminar)
    parser.add_argument(
        "--prefill-tokens",
        default=None,
        help="Comma-separated prefill tokens (e.g., '1,2,3,4,5')"
    )
    parser.add_argument(
        "--num-decode-tokens",
        type=int,
        default=None,
        help="Number of tokens to generate during decode (e.g., 3)"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        default="pytorch_incremental_snapshots",
        help="Output directory (default: pytorch_incremental_snapshots)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Determine mode
    if args.prefill_tokens is not None and args.num_decode_tokens is not None:
        # Mode 2: Prefill + decode
        prefill_tokens = [int(t.strip()) for t in args.prefill_tokens.split(',')]
        snapshots, sampled_tokens = incremental_decode_with_prefill(
            args.model,
            prefill_tokens,
            args.num_decode_tokens,
            verbose=args.verbose
        )
    elif args.tokens is not None:
        # Mode 1: Process all tokens
        token_sequence = [int(t.strip()) for t in args.tokens.split(',')]
        snapshots, sampled_tokens = incremental_decode_with_cache(
            args.model,
            token_sequence,
            verbose=args.verbose
        )
    else:
        # Default: use tokens mode with default sequence
        token_sequence = [1, 2, 3, 4, 5, 6]
        if args.verbose:
            print(f"Using default token sequence: {token_sequence}")
        snapshots, sampled_tokens = incremental_decode_with_cache(
            args.model,
            token_sequence,
            verbose=args.verbose
        )
    
    # Save to disk
    save_incremental_snapshots(
        snapshots,
        sampled_tokens,
        Path(args.output_dir),
        verbose=args.verbose
    )
    
    # Save model weights for verification
    save_model_weights(
        args.model,
        Path(args.output_dir),
        verbose=args.verbose
    )
    
    print(f"\n✓ Complete! Snapshots saved to {args.output_dir}/")
    print(f"  Sampled tokens: {sampled_tokens}")

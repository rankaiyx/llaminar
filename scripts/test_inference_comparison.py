#!/usr/bin/env python3
"""
Test inference comparison between PyTorch and Llaminar for fp16/fp32 models.

This script runs three simple English questions through both systems and compares
the generated text outputs to verify they produce identical results.

@author David Sanftenberg
"""

import os
import sys
import subprocess
import argparse
import re
from pathlib import Path
from typing import List, Tuple

# Add parent directories to path
script_dir = Path(__file__).parent.absolute()
workspace_dir = script_dir.parent.absolute()
python_dir = workspace_dir / "python"

for path_to_add in [str(python_dir), str(workspace_dir)]:
    if path_to_add not in sys.path:
        sys.path.insert(0, path_to_add)


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def run_pytorch_inference(model_path: str, prompt: str, max_tokens: int = 20) -> str:
    """
    Run inference using PyTorch reference implementation via llama.cpp.
    
    Args:
        model_path: Path to the model file
        prompt: Input prompt text
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated text output
    """
    print(f"{Colors.BLUE}[PyTorch/llama.cpp] Running inference...{Colors.END}")
    
    # Use llama.cpp for PyTorch-equivalent inference
    llamacpp_bin = workspace_dir / "llama.cpp" / "build" / "bin" / "llama-cli"
    if not llamacpp_bin.exists():
        llamacpp_bin = workspace_dir / "llama.cpp" / "llama-cli"
    if not llamacpp_bin.exists():
        raise RuntimeError(f"llama.cpp binary not found. Expected at {llamacpp_bin}")
    
    cmd = [
        str(llamacpp_bin),
        "-m", str(model_path),
        "-p", prompt,
        "-n", str(max_tokens),
        "--temp", "0.0",  # Deterministic (greedy)
        "--top-k", "1",
        "-ngl", "0",  # CPU only for consistency
    ]
    
    print(f"  Prompt: '{prompt}'")
    print(f"  Command: {' '.join(cmd[:8])}...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(workspace_dir)
        )
        
        if result.returncode != 0:
            print(f"{Colors.RED}Error running llama.cpp:{Colors.END}")
            print(f"STDERR: {result.stderr[:500]}")
            raise RuntimeError(f"llama.cpp exited with code {result.returncode}")
        
        # Parse output - llama.cpp prints the full conversation
        output = result.stdout
        
        # Extract just the generated part (after the prompt)
        # llama.cpp output typically looks like:
        # <prompt text><generated text>
        # We need to extract everything after the prompt
        
        # Find where generation starts - look for the prompt end
        if prompt in output:
            gen_start = output.find(prompt) + len(prompt)
            generated = output[gen_start:].strip()
        else:
            # Fallback: try to find common markers
            lines = output.split('\n')
            # Skip log lines and get the actual output
            generated_lines = [l for l in lines if l and not l.startswith('[') and not l.startswith('llama')]
            generated = ' '.join(generated_lines).strip()
        
        # Clean up common artifacts
        generated = generated.replace('[end of text]', '').strip()
        
        print(f"  Generated: '{generated[:100]}{'...' if len(generated) > 100 else ''}'")
        return generated
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("llama.cpp inference timed out after 60 seconds")


def run_llaminar_inference(model_path: str, prompt: str, max_tokens: int = 20) -> str:
    """
    Run inference using Llaminar C++ implementation.
    
    Args:
        model_path: Path to the model file
        prompt: Input prompt text
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated text output
    """
    print(f"{Colors.BLUE}[Llaminar] Running inference...{Colors.END}")
    
    # Build command
    llaminar_bin = workspace_dir / "build" / "llaminar"
    if not llaminar_bin.exists():
        raise RuntimeError(f"Llaminar binary not found: {llaminar_bin}")
    
    cmd = [
        str(llaminar_bin),
        "-m", str(model_path),
        "-p", prompt,
        "-n", str(max_tokens),
        "--temp", "0.0",  # Deterministic sampling
        "--top-k", "1",   # Greedy decoding
    ]
    
    print(f"  Prompt: '{prompt}'")
    print(f"  Command: {' '.join(cmd[:8])}...")
    
    # Run inference
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(workspace_dir)
        )
        
        if result.returncode != 0:
            print(f"{Colors.RED}Error running Llaminar:{Colors.END}")
            print(f"STDERR: {result.stderr[:500]}")
            raise RuntimeError(f"Llaminar exited with code {result.returncode}")
        
        # Parse output
        output = result.stdout
        
        # Extract generated text (after the prompt)
        if prompt in output:
            gen_start = output.find(prompt) + len(prompt)
            generated = output[gen_start:].strip()
        else:
            # Fallback parsing
            lines = output.split('\n')
            generated_lines = [l for l in lines if l and not l.startswith('[') and 'Llaminar' not in l]
            generated = ' '.join(generated_lines).strip()
        
        # Clean up artifacts
        generated = generated.replace('[end of text]', '').strip()
        
        print(f"  Generated: '{generated[:100]}{'...' if len(generated) > 100 else ''}'")
        return generated
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Llaminar inference timed out after 60 seconds")


def compare_outputs(pytorch_out: str, llaminar_out: str, question: str) -> bool:
    """
    Compare PyTorch and Llaminar outputs.
    
    Args:
        pytorch_out: PyTorch generated text
        llaminar_out: Llaminar generated text
        question: The original question
        
    Returns:
        True if outputs match, False otherwise
    """
    # Strip whitespace for comparison
    pytorch_clean = pytorch_out.strip()
    llaminar_clean = llaminar_out.strip()
    
    match = pytorch_clean == llaminar_clean
    
    print(f"\n{Colors.BOLD}Comparison for: '{question}'{Colors.END}")
    print(f"  PyTorch:  '{pytorch_clean}'")
    print(f"  Llaminar: '{llaminar_clean}'")
    
    if match:
        print(f"  {Colors.GREEN}✓ MATCH{Colors.END}")
    else:
        print(f"  {Colors.RED}✗ MISMATCH{Colors.END}")
        # Show character-level difference
        min_len = min(len(pytorch_clean), len(llaminar_clean))
        for i in range(min_len):
            if pytorch_clean[i] != llaminar_clean[i]:
                print(f"  First difference at position {i}:")
                print(f"    PyTorch:  '{pytorch_clean[max(0,i-10):i+20]}'")
                print(f"    Llaminar: '{llaminar_clean[max(0,i-10):i+20]}'")
                break
        if len(pytorch_clean) != len(llaminar_clean):
            print(f"  Length difference: PyTorch={len(pytorch_clean)}, Llaminar={len(llaminar_clean)}")
    
    return match


def main():
    parser = argparse.ArgumentParser(
        description="Compare PyTorch and Llaminar inference outputs"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="models/qwen2.5-0.5b-instruct-f32.gguf",
        help="Path to fp32 model file (default: models/qwen2.5-0.5b-instruct-f32.gguf)"
    )
    parser.add_argument(
        "-n", "--max-tokens",
        type=int,
        default=20,
        help="Maximum tokens to generate per question (default: 20)"
    )
    
    args = parser.parse_args()
    
    # Check model exists
    model_path = workspace_dir / args.model
    if not model_path.exists():
        print(f"{Colors.RED}Error: Model not found: {model_path}{Colors.END}")
        print(f"Please provide a valid fp32 model file.")
        return 1
    
    # Test questions
    questions = [
        "What is the capital of France?",
        "What color is the sky?",
        "How many days are in a week?",
    ]
    
    print(f"{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}Inference Comparison Test: PyTorch vs Llaminar{Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"Model: {model_path}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Questions: {len(questions)}")
    print()
    
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"{Colors.BOLD}{'='*80}{Colors.END}")
        print(f"{Colors.BOLD}Question {i}/{len(questions)}{Colors.END}")
        print(f"{Colors.BOLD}{'='*80}{Colors.END}")
        
        try:
            # Run PyTorch
            pytorch_output = run_pytorch_inference(str(model_path), question, args.max_tokens)
            print()
            
            # Run Llaminar
            llaminar_output = run_llaminar_inference(str(model_path), question, args.max_tokens)
            print()
            
            # Compare
            match = compare_outputs(pytorch_output, llaminar_output, question)
            results.append((question, match, pytorch_output, llaminar_output))
            print()
            
        except Exception as e:
            print(f"{Colors.RED}Error processing question '{question}': {e}{Colors.END}")
            results.append((question, False, "", ""))
            print()
    
    # Summary
    print(f"{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}SUMMARY{Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}")
    
    matches = sum(1 for _, match, _, _ in results if match)
    total = len(results)
    
    print(f"\nResults: {matches}/{total} questions matched")
    print()
    
    for i, (question, match, pytorch_out, llaminar_out) in enumerate(results, 1):
        status = f"{Colors.GREEN}✓{Colors.END}" if match else f"{Colors.RED}✗{Colors.END}"
        print(f"{status} Q{i}: {question}")
        if not match:
            print(f"     PyTorch:  '{pytorch_out[:60]}{'...' if len(pytorch_out) > 60 else ''}'")
            print(f"     Llaminar: '{llaminar_out[:60]}{'...' if len(llaminar_out) > 60 else ''}'")
    
    print()
    
    if matches == total:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ ALL TESTS PASSED{Colors.END}")
        print(f"{Colors.GREEN}PyTorch and Llaminar produce identical outputs!{Colors.END}")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ TESTS FAILED{Colors.END}")
        print(f"{Colors.RED}{matches}/{total} questions matched{Colors.END}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Generate text reference using PyTorch for inference parity testing.

This script generates text from a prompt using PyTorch with GGUF model loading
(including dequantization) and saves the generated token IDs for comparison 
with Llaminar's C++ implementation.

CRITICAL: This loads the EXACT SAME GGUF FILE (with quantization) as Llaminar,
ensuring a valid apples-to-apples comparison.

@author David Sanftenberg
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np

# Add parent directories to path
script_dir = Path(__file__).parent.absolute()
python_dir = script_dir.parent.absolute()
workspace_dir = python_dir.parent.absolute()

for path_to_add in [str(python_dir), str(workspace_dir)]:
    if path_to_add not in sys.path:
        sys.path.insert(0, path_to_add)

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError as e:
    print(f"Error: {e}")
    print("Please install required packages: pip install torch transformers")
    sys.exit(1)

# Import our GGUF loader
try:
    from python.reference.loaders.gguf_loader import GGUFLoader
    from python.reference.qwen import QwenReferenceModel
    HAS_GGUF_LOADER = True
except ImportError as e:
    print(f"Warning: Could not import GGUF loader: {e}")
    HAS_GGUF_LOADER = False


def generate_text_reference(
    model_path: str,
    prompt: str,
    max_new_tokens: int = 20,
    temperature: float = 0.0,
    output_path: str = None
) -> Dict:
    """
    Generate text using PyTorch model loaded from GGUF file with greedy decoding.
    
    CRITICAL: This loads the exact GGUF file (with quantization intact) that
    Llaminar uses, dequantizes it, and runs inference. This ensures we're comparing
    the same model at the same precision level.
    
    Args:
        model_path: Path to GGUF model file (e.g., qwen2.5-0.5b-instruct-q4_0.gguf)
        prompt: Input text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 = greedy/deterministic)
        output_path: Where to save results (JSON)
    
    Returns:
        Dictionary with prompt_tokens, generated_tokens, and generated_text
    """
    print(f"\n{'='*80}")
    print("PyTorch Text Generation Reference (GGUF Loader)")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"Prompt: '{prompt}'")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    print()
    
    if not HAS_GGUF_LOADER:
        print("ERROR: GGUF loader not available. Cannot load GGUF file.")
        print("Please ensure python/reference/loaders/ is in the path.")
        sys.exit(1)
    
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)
    
    # Determine architecture from GGUF metadata
    print(f"Loading GGUF file: {model_path}")
    loader = GGUFLoader(model_path, verbose=True)
    
    # Parse GGUF to extract metadata (needed for tokenizer)
    from python.reference.loaders.gguf_parser import GGUFParser
    parser = GGUFParser(model_path)
    parser.parse()
    metadata = parser.metadata
    parser.close()
    
    # Load model
    config, state_dict = loader.load()
    
    # Determine model architecture
    if hasattr(config, 'architectures') and config.architectures:
        arch_name = config.architectures[0]
    elif hasattr(config, 'model_type'):
        arch_name = config.model_type
    else:
        arch_name = "qwen2"  # Default for Qwen models
    
    print(f"Architecture: {arch_name}")
    
    # Load model with dequantized weights
    print(f"Creating PyTorch model from dequantized GGUF weights...")
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict, strict=False)  # strict=False for potential missing keys
    model.eval()
    
    print(f"✓ Model loaded from GGUF ({model.num_parameters():,} parameters)")
    
    # Load tokenizer directly from GGUF metadata
    # GGUF tokenizer is used for DECODING output tokens (ensures vocab matches model)
    # HuggingFace tokenizer is used for ENCODING input (simpler, already implemented)
    print(f"Loading tokenizers...")
    
    # Load GGUF tokenizer for decoding
    try:
        from python.reference.loaders.gguf_tokenizer import GGUFTokenizer
        gguf_tokenizer = GGUFTokenizer.from_gguf_metadata(metadata)
        print(f"✓ GGUF tokenizer loaded for decoding (vocab_size={gguf_tokenizer.vocab_size})")
    except Exception as e:
        print(f"⚠ Failed to load GGUF tokenizer: {e}")
        gguf_tokenizer = None
    
    # Load HuggingFace tokenizer for encoding (and fallback decoding)
    tokenizer_name = "Qwen/Qwen2.5-0.5B-Instruct" if "qwen" in model_path.lower() else "meta-llama/Llama-2-7b-hf"
    print(f"  Loading HuggingFace tokenizer for encoding: {tokenizer_name}")
    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    
    # Tokenize prompt (always use HuggingFace for encoding)
    input_ids = hf_tokenizer(prompt, return_tensors="pt")['input_ids']
    prompt_tokens = input_ids[0].tolist()
    
    print(f"\nPrompt tokens: {prompt_tokens}")
    print(f"Prompt length: {len(prompt_tokens)} tokens")
    
    # Use HuggingFace EOS token for generation
    eos_token_id = hf_tokenizer.eos_token_id
    
    # Generate with deterministic sampling
    with torch.no_grad():
        if temperature == 0.0:
            # Greedy decoding (deterministic)
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy
                pad_token_id=eos_token_id
            )
        else:
            # Temperature sampling
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                top_k=0,
                pad_token_id=eos_token_id
            )
    
    # Extract generated tokens (excluding prompt)
    full_output = output[0].tolist()
    generated_tokens = full_output[len(prompt_tokens):]
    
    # Decode generated text
    # Use GGUF tokenizer if available (correct vocab), otherwise HuggingFace
    if gguf_tokenizer is not None:
        print(f"\n[Using GGUF tokenizer for decoding - vocab size: {gguf_tokenizer.vocab_size}]")
        generated_text = gguf_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        full_text = gguf_tokenizer.decode(full_output, skip_special_tokens=True)
    else:
        print(f"\n[Using HuggingFace tokenizer for decoding - may cause vocab mismatch!]")
        generated_text = hf_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        full_text = hf_tokenizer.decode(full_output, skip_special_tokens=True)
    
    print(f"\nGenerated tokens: {generated_tokens}")
    print(f"Generated {len(generated_tokens)} tokens")
    print(f"\nFull output: '{full_text}'")
    print(f"Generated text: '{generated_text}'")
    
    # Prepare result
    result = {
        "model_path": model_path,
        "loaded_from_gguf": True,
        "architecture": arch_name,
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "generated_text": generated_text,
        "full_output_tokens": full_output,
        "full_output_text": full_text,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature
    }
    
    # Save to file if requested
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n✓ Saved reference to: {output_file}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate text reference using PyTorch"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to GGUF model file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Input prompt text"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy/deterministic)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    result = generate_text_reference(
        args.model,
        args.prompt,
        args.max_tokens,
        args.temperature,
        args.output
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

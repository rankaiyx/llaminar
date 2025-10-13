#!/usr/bin/env python3
"""
Simple inference test for Llaminar.

Tests that Llaminar can successfully run inference on simple English questions
and produce reasonable outputs.

@author David Sanftenberg
"""

import subprocess
import sys
from pathlib import Path

workspace_dir = Path(__file__).parent.parent.absolute()

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def run_llaminar_test(model_path: str, prompt: str, max_tokens: int = 20) -> tuple[bool, str]:
    """Run a single inference test."""
    print(f"\n{Colors.BOLD}Testing:{Colors.END} '{prompt}'")
    
    llaminar_bin = workspace_dir / "build" / "llaminar"
    if not llaminar_bin.exists():
        return False, f"Llaminar binary not found: {llaminar_bin}"
    
    cmd = [
        str(llaminar_bin),
        "-m", str(model_path),
        "-p", prompt,
        "--predict", str(max_tokens),
        "--temperature", "0.0",
        "--top-k", "1",
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            return False, f"Exit code {result.returncode}: {result.stderr[:200]}"
        
        output = result.stdout
        
        # Check for errors in output
        if "Error" in output or "error" in output.lower():
            if "error" in output.lower() and "error" not in prompt.lower():
                return False, f"Error in output: {output[:200]}"
        
        # Check that we got some output
        if len(output.strip()) < len(prompt):
            return False, f"Output too short ({len(output)} chars)"
        
        # Extract just the visible output (skip log lines)
        visible_lines = []
        for line in output.split('\n'):
            # Skip log lines that start with [ or common prefixes
            if (not line.startswith('[') and 
                not line.startswith('GGUF') and 
                not line.startswith('Loading') and
                not line.strip().startswith('✓') and
                line.strip()):
                visible_lines.append(line)
        
        visible_output = '\n'.join(visible_lines)
        
        print(f"{Colors.BLUE}Generated:{Colors.END} {visible_output[:200]}")
        return True, visible_output
        
    except subprocess.TimeoutExpired:
        return False, "Timeout after 60 seconds"
    except Exception as e:
        return False, f"Exception: {str(e)}"


def main():
    model_path = workspace_dir / "models" / "qwen2.5-0.5b-instruct-fp16.gguf"
    
    if not model_path.exists():
        print(f"{Colors.RED}Model not found: {model_path}{Colors.END}")
        return 1
    
    print(f"{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}Llaminar Simple Inference Test{Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"Model: {model_path}")
    print(f"Testing 3 simple questions...")
    
    questions = [
        "What is 2+2?",
        "What color is grass?",
        "How many days in a week?",
    ]
    
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n{Colors.BOLD}[{i}/{len(questions)}]{Colors.END}")
        success, output = run_llaminar_test(str(model_path), question, max_tokens=15)
        results.append((question, success, output))
        
        if success:
            print(f"{Colors.GREEN}✓ PASSED{Colors.END}")
        else:
            print(f"{Colors.RED}✗ FAILED: {output}{Colors.END}")
    
    # Summary
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}SUMMARY{Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}\n")
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for question, success, output in results:
        status = f"{Colors.GREEN}✓{Colors.END}" if success else f"{Colors.RED}✗{Colors.END}"
        print(f"{status} {question}")
        if not success:
            print(f"    Error: {output[:80]}")
    
    print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print(f"{Colors.GREEN}✓ ALL TESTS PASSED{Colors.END}")
        return 0
    else:
        print(f"{Colors.RED}✗ {total - passed} TEST(S) FAILED{Colors.END}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Analyze effective K/V dumps from AttentionComputeStage.

Usage:
    # Run with TQ4 first:
    LLAMINAR_DUMP_EFFECTIVE_KV=1 ctest --test-dir build_v2_integration \
        -R "V2_Integration_Parity_Qwen3_TQ4_Analysis" --verbose
    mv /tmp/effective_kv_dump /tmp/effective_kv_dump_tq4

    # Then run with FP16:
    LLAMINAR_DUMP_EFFECTIVE_KV=1 ctest --test-dir build_v2_integration \
        -R "V2_Integration_Parity_Qwen3_FP16_something" --verbose
    mv /tmp/effective_kv_dump /tmp/effective_kv_dump_fp16

    # Then analyze:
    python3 scripts/analyze_effective_kv.py /tmp/effective_kv_dump_tq4

    # Or compare TQ4 vs FP16:
    python3 scripts/analyze_effective_kv.py /tmp/effective_kv_dump_tq4 --ref /tmp/effective_kv_dump_fp16
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def load_meta(meta_path: Path) -> dict:
    """Load a key=value metadata file."""
    meta = {}
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()
    return meta


def load_fp32_bin(bin_path: Path, n_elements: int) -> np.ndarray:
    """Load a binary file of FP32 values."""
    data = np.fromfile(str(bin_path), dtype=np.float32)
    if data.size != n_elements:
        print(f"  WARNING: Expected {n_elements} elements, got {data.size} in {bin_path.name}")
    return data


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a.astype(np.float64), b.astype(np.float64))
    na = np.linalg.norm(a.astype(np.float64))
    nb = np.linalg.norm(b.astype(np.float64))
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(dot / (na * nb))


def per_head_cosine(a: np.ndarray, b: np.ndarray, n_heads: int, head_dim: int) -> list:
    """Compute per-head cosine similarity for KV data [rows, n_heads*head_dim]."""
    results = []
    for h in range(n_heads):
        a_h = a[:, h * head_dim : (h + 1) * head_dim].flatten()
        b_h = b[:, h * head_dim : (h + 1) * head_dim].flatten()
        results.append(cosine_sim(a_h, b_h))
    return results


def per_row_cosine(a: np.ndarray, b: np.ndarray) -> list:
    """Compute per-row cosine similarity."""
    results = []
    for r in range(a.shape[0]):
        results.append(cosine_sim(a[r], b[r]))
    return results


def analyze_tensor(name: str, data: np.ndarray, rows: int, cols: int):
    """Print detailed statistics about a tensor."""
    reshaped = data.reshape(rows, cols) if data.size == rows * cols else data
    print(f"\n  {name}: shape=({rows}, {cols})")
    print(f"    min={data.min():.6f}  max={data.max():.6f}  mean={data.mean():.6f}")
    print(f"    std={data.std():.6f}  norm={np.linalg.norm(data):.4f}")
    nan_count = np.isnan(data).sum()
    inf_count = np.isinf(data).sum()
    zero_count = (data == 0.0).sum()
    if nan_count > 0:
        print(f"    *** {nan_count} NaN values! ***")
    if inf_count > 0:
        print(f"    *** {inf_count} Inf values! ***")
    if zero_count > data.size * 0.5:
        print(f"    *** {zero_count}/{data.size} zeros ({100*zero_count/data.size:.1f}%) ***")

    # Per-row norms (first and last 3)
    if rows > 1:
        norms = np.linalg.norm(reshaped, axis=1)
        show_rows = min(3, rows)
        print(f"    row norms[0:{show_rows}]: {norms[:show_rows]}")
        if rows > show_rows:
            print(f"    row norms[-{show_rows}:]: {norms[-show_rows:]}")

    return reshaped


def analyze_tq4_raw(dump_dir: Path, meta: dict):
    """Analyze raw TQ4 cache blocks."""
    cache_meta_path = dump_dir / "K_cache_meta.txt"
    cache_bin_path = dump_dir / "K_cache_tq4.bin"
    if not cache_bin_path.exists():
        return

    cmeta = load_meta(cache_meta_path)
    bpr = int(cmeta["blocks_per_row"])
    bb = int(cmeta["block_bytes"])
    hd = int(cmeta["head_dim"])
    rows = int(cmeta["rows"])

    raw = np.fromfile(str(cache_bin_path), dtype=np.uint8)
    print(f"\n  TQ4 Raw Cache K: {rows} rows × {bpr} blocks × {bb} bytes = {raw.size} bytes")

    # Parse TQ4 blocks: [float norm, float residual_norm, mse_indices..., high_bits...]
    for r in [0, rows - 1]:
        print(f"    Row {r}:")
        for h in range(min(3, bpr)):
            offset = (r * bpr + h) * bb
            block = raw[offset : offset + bb]
            norm = struct.unpack("f", block[0:4])[0]
            res_norm = struct.unpack("f", block[4:8])[0]
            mse_bytes = block[8 : 8 + hd * 3 // 8]
            high_bytes = block[8 + hd * 3 // 8 :]
            # Check if block is all zeros (uninitialized)
            is_zero = (block == 0).all()
            print(
                f"      head[{h}]: norm={norm:.6f}  res_norm={res_norm:.6f}"
                f"  mse_nonzero={np.count_nonzero(mse_bytes)}"
                f"  high_nonzero={np.count_nonzero(high_bytes)}"
                f"  {'*** ALL ZEROS ***' if is_zero else ''}"
            )


def analyze_single_dump(dump_dir: Path, ref_dir: Path = None):
    """Analyze a single iteration's dump."""
    meta = load_meta(dump_dir / "meta.txt")

    layer = int(meta["layer"])
    iteration = int(meta["iteration"])
    seq_len = int(meta["seq_len"])
    kv_len = int(meta["kv_len"])
    n_heads = int(meta["n_heads"])
    n_kv_heads = int(meta["n_kv_heads"])
    head_dim = int(meta["head_dim"])
    k_type = meta.get("K_type", "unknown")
    v_type = meta.get("V_type", "unknown")
    k_dequanted = meta.get("K_is_dequanted", "0") == "1"
    v_dequanted = meta.get("V_is_dequanted", "0") == "1"

    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    print(f"\n{'='*80}")
    print(f"  Layer {layer}  Iteration {iteration}")
    print(f"  seq_len={seq_len}  kv_len={kv_len}  head_dim={head_dim}")
    print(f"  n_heads={n_heads}  n_kv_heads={n_kv_heads}")
    print(f"  K_type={k_type}  V_type={v_type}")
    print(f"  K_is_dequanted={k_dequanted}  V_is_dequanted={v_dequanted}")
    print(f"{'='*80}")

    # Load tensors
    tensors = {}
    for name, expected_elems, rows, cols in [
        ("Q", seq_len * q_dim, seq_len, q_dim),
        ("K_effective", kv_len * kv_dim, kv_len, kv_dim),
        ("V_effective", kv_len * kv_dim, kv_len, kv_dim),
    ]:
        bin_path = dump_dir / f"{name}.bin"
        if bin_path.exists():
            data = load_fp32_bin(bin_path, expected_elems)
            tensors[name] = analyze_tensor(name, data, rows, cols)
        else:
            print(f"\n  {name}: NOT FOUND")

    # Analyze TQ4 raw blocks
    analyze_tq4_raw(dump_dir, meta)

    # Per-head analysis of K
    if "K_effective" in tensors:
        k = tensors["K_effective"]
        print(f"\n  === Per-Head K Analysis (kv_len={kv_len}) ===")
        for h in range(n_kv_heads):
            k_h = k[:, h * head_dim : (h + 1) * head_dim]
            print(
                f"    head[{h}]: norm={np.linalg.norm(k_h):.4f}"
                f"  mean={k_h.mean():.6f}"
                f"  std={k_h.std():.6f}"
                f"  min={k_h.min():.6f}"
                f"  max={k_h.max():.6f}"
                f"  zeros={np.sum(k_h == 0)}/{k_h.size}"
            )

    # Compare with reference if available
    if ref_dir is not None:
        ref_meta_path = ref_dir / "meta.txt"
        if ref_meta_path.exists():
            ref_meta = load_meta(ref_meta_path)
            ref_kv_len = int(ref_meta["kv_len"])

            print(f"\n  === COMPARISON vs Reference ===")
            print(f"  Ref kv_len={ref_kv_len}  This kv_len={kv_len}")

            compare_len = min(kv_len, ref_kv_len)

            for name, cols, nh, hd in [
                ("K_effective", kv_dim, n_kv_heads, head_dim),
                ("V_effective", kv_dim, n_kv_heads, head_dim),
            ]:
                ref_path = ref_dir / f"{name}.bin"
                this_path = dump_dir / f"{name}.bin"
                if not ref_path.exists() or not this_path.exists():
                    continue

                ref_data = load_fp32_bin(ref_path, ref_kv_len * cols).reshape(ref_kv_len, cols)
                this_data = load_fp32_bin(this_path, kv_len * cols).reshape(kv_len, cols)

                # Compare overlapping rows
                ref_clip = ref_data[:compare_len]
                this_clip = this_data[:compare_len]

                overall_cos = cosine_sim(ref_clip.flatten(), this_clip.flatten())
                print(f"\n  {name} overall cosine: {overall_cos:.6f}")

                # Per-head
                head_cos = per_head_cosine(ref_clip, this_clip, nh, hd)
                for h, c in enumerate(head_cos):
                    print(f"    head[{h}]: cosine={c:.6f}")

                # Per-row
                row_cos = per_row_cosine(ref_clip, this_clip)
                print(f"  {name} per-row cosine (first 5, last 5):")
                for r in range(min(5, compare_len)):
                    print(f"    row[{r}]: {row_cos[r]:.6f}")
                if compare_len > 5:
                    for r in range(max(5, compare_len - 5), compare_len):
                        print(f"    row[{r}]: {row_cos[r]:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze effective K/V dumps")
    parser.add_argument("dump_dir", type=Path, help="Path to effective_kv_dump directory")
    parser.add_argument("--ref", type=Path, default=None, help="Reference dump dir for comparison")
    parser.add_argument("--layer", type=int, default=0, help="Layer to analyze (default: 0)")
    parser.add_argument("--iter", type=int, default=None, help="Specific iteration (default: all)")
    args = parser.parse_args()

    if not args.dump_dir.exists():
        print(f"ERROR: {args.dump_dir} does not exist")
        sys.exit(1)

    # Find all dump subdirectories
    pattern = f"layer{args.layer}_iter*"
    dump_dirs = sorted(args.dump_dir.glob(pattern))

    if not dump_dirs:
        # Try without layer prefix
        dump_dirs = sorted(d for d in args.dump_dir.iterdir() if d.is_dir())

    if not dump_dirs:
        print(f"ERROR: No dump directories found in {args.dump_dir}")
        sys.exit(1)

    print(f"Found {len(dump_dirs)} dump(s) in {args.dump_dir}")

    for dd in dump_dirs:
        if args.iter is not None:
            if f"iter{args.iter}" not in dd.name:
                continue

        ref_dd = None
        if args.ref:
            ref_dd = args.ref / dd.name
            if not ref_dd.exists():
                ref_dd = None

        analyze_single_dump(dd, ref_dd)

    print(f"\n{'='*80}")
    print("Analysis complete.")


if __name__ == "__main__":
    main()

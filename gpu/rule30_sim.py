"""
GPU-accelerated Rule 30 simulation using CuPy bit-packing.

Rule 30: new[i] = left[i] XOR (center[i] OR right[i])
Equivalently on packed uint64 words: new = left ^ (center | right)

Stores tape as array of uint64, each holding 64 cells.
"""
import sys
import time
import os
import json
import cupy as cp
import numpy as np
from tqdm import tqdm

# Rule 30 kernel operating on uint64-packed tape
rule30_kernel = cp.RawKernel(r'''
extern "C" __global__
void rule30_step(const unsigned long long* tape, unsigned long long* out, int n_words) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_words) return;

    unsigned long long center = tape[idx];
    unsigned long long prev_word = (idx > 0) ? tape[idx - 1] : 0ULL;
    unsigned long long next_word = (idx < n_words - 1) ? tape[idx + 1] : 0ULL;

    // left neighbor: shift tape right by 1 bit position
    unsigned long long left_word = (center >> 1) | (prev_word << 63);
    // right neighbor: shift tape left by 1 bit position
    unsigned long long right_word = (center << 1) | (next_word >> 63);

    // Rule 30: left XOR (center OR right)
    out[idx] = left_word ^ (center | right_word);
}
''', 'rule30_step')


def simulate(n_cells, n_steps, extract_center=False, center_out_path=None):
    """Run Rule 30 simulation on GPU with tqdm progress bar."""
    n_words = (n_cells + 63) // 64
    n_cells = n_words * 64
    center_word_idx = n_words // 2
    center_bit_idx = 32

    print(f"Tape: {n_cells:,} cells = {n_words:,} uint64 words")
    print(f"Steps: {n_steps:,}")
    print(f"GPU memory for tape: {n_words * 8 * 2 / 1024 / 1024:.1f} MB (double-buffered)")

    tape_a = cp.zeros(n_words, dtype=cp.uint64)
    tape_b = cp.zeros(n_words, dtype=cp.uint64)
    tape_a[center_word_idx] = cp.uint64(1 << center_bit_idx)

    if extract_center:
        center_bytes = np.zeros((n_steps + 7) // 8, dtype=np.uint8)

    block_size = 256
    grid_size = (n_words + block_size - 1) // block_size

    mempool = cp.get_default_memory_pool()

    cp.cuda.Stream.null.synchronize()
    print(f"Initial center word: {int(tape_a[center_word_idx]):#018x}")

    start = time.perf_counter()
    current = tape_a
    next_buf = tape_b

    pbar = tqdm(range(n_steps), desc="Rule 30", unit="step", unit_scale=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for step in pbar:
        if extract_center:
            bit = int((current[center_word_idx] >> center_bit_idx) & 1)
            byte_idx = step // 8
            bit_idx = step % 8
            if bit:
                center_bytes[byte_idx] |= (1 << bit_idx)

        rule30_kernel((grid_size,), (block_size,), (current, next_buf, n_words))
        current, next_buf = next_buf, current

    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    pbar.close()

    vram_after = mempool.used_bytes() / 1024 / 1024
    steps_per_sec = n_steps / elapsed
    cells_per_sec = n_cells * n_steps / elapsed

    results = {
        "n_cells": n_cells,
        "n_steps": n_steps,
        "elapsed_seconds": round(elapsed, 2),
        "steps_per_second": round(steps_per_sec, 1),
        "cells_per_second": round(cells_per_sec, 1),
        "gcells_per_second": round(cells_per_sec / 1e9, 3),
        "vram_mb": round(vram_after, 1),
    }

    print(f"\n=== Results ===")
    print(f"Time:        {elapsed:.2f} s")
    print(f"Steps/sec:   {steps_per_sec:,.0f}")
    print(f"Cells/sec:   {cells_per_sec:,.0f} ({cells_per_sec/1e9:.3f} Gcells/s)")
    print(f"VRAM used:   {vram_after:.1f} MB")

    if extract_center and center_out_path:
        os.makedirs(os.path.dirname(center_out_path), exist_ok=True)
        with open(center_out_path, 'wb') as f:
            f.write(center_bytes)
        print(f"Center column saved: {center_out_path} ({len(center_bytes):,} bytes = {n_steps:,} bits)")
        results["center_col_file"] = center_out_path
        results["center_col_bits"] = n_steps

        # Verify first 20 bits against CPU reference
        expected = [1,1,0,1,1,1,0,0,1,1,0,0,0,1,0,1,1,0,0,1]
        actual = [(center_bytes[i // 8] >> (i % 8)) & 1 for i in range(min(20, n_steps))]
        match = list(actual) == expected[:len(actual)]
        print(f"First 20 bits: {list(actual)}")
        print(f"Expected:      {expected}")
        print(f"Verification:  {'PASS' if match else 'FAIL'}")
        results["verification_passed"] = match

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPU Rule 30 simulation")
    parser.add_argument("--cells", type=int, default=1_000_000, help="Tape width in cells")
    parser.add_argument("--steps", type=int, default=10_000, help="Number of steps")
    parser.add_argument("--center", action="store_true", help="Extract center column")
    parser.add_argument("--center-out", type=str, default=None, help="Path to save center column bits")
    parser.add_argument("--json-out", type=str, default=None, help="Save results as JSON")
    args = parser.parse_args()

    results = simulate(
        n_cells=args.cells,
        n_steps=args.steps,
        extract_center=args.center,
        center_out_path=args.center_out,
    )

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) if os.path.dirname(args.json_out) else ".", exist_ok=True)
        with open(args.json_out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.json_out}")

    return results


if __name__ == "__main__":
    main()

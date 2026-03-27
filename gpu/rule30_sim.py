"""
GPU-accelerated Rule 30 simulation using CuPy bit-packing.

Rule 30: new[i] = left[i] XOR (center[i] OR right[i])
Stores tape as array of uint64, each holding 64 cells.

Center column extraction is batched on-GPU to avoid per-step sync.
"""
import sys
import time
import os
import json
import cupy as cp
import numpy as np
from tqdm import tqdm

# Rule 30 step + center bit extraction in one kernel launch
# Thread 0 also extracts the center bit into the output buffer
rule30_with_center_kernel = cp.RawKernel(r'''
extern "C" __global__
void rule30_step_center(
    const unsigned long long* tape,
    unsigned long long* out,
    int n_words,
    int center_word_idx,
    int center_bit_idx,
    unsigned char* center_out,  // packed bit output buffer
    int step                    // current step number
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_words) return;

    unsigned long long center = tape[idx];

    // Extract center bit (only thread handling the center word)
    if (idx == center_word_idx) {
        unsigned char bit = (center >> center_bit_idx) & 1ULL;
        int byte_idx = step / 8;
        int bit_idx = step % 8;
        if (bit) {
            atomicOr((unsigned int*)&center_out[byte_idx & ~3],
                     ((unsigned int)bit) << (8 * (byte_idx & 3) + bit_idx));
        }
    }

    unsigned long long prev_word = (idx > 0) ? tape[idx - 1] : 0ULL;
    unsigned long long next_word = (idx < n_words - 1) ? tape[idx + 1] : 0ULL;

    unsigned long long left_word = (center >> 1) | (prev_word << 63);
    unsigned long long right_word = (center << 1) | (next_word >> 63);

    out[idx] = left_word ^ (center | right_word);
}
''', 'rule30_step_center')

# Simple Rule 30 kernel (no center extraction)
rule30_kernel = cp.RawKernel(r'''
extern "C" __global__
void rule30_step(const unsigned long long* tape, unsigned long long* out, int n_words) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_words) return;

    unsigned long long center = tape[idx];
    unsigned long long prev_word = (idx > 0) ? tape[idx - 1] : 0ULL;
    unsigned long long next_word = (idx < n_words - 1) ? tape[idx + 1] : 0ULL;

    unsigned long long left_word = (center >> 1) | (prev_word << 63);
    unsigned long long right_word = (center << 1) | (next_word >> 63);

    out[idx] = left_word ^ (center | right_word);
}
''', 'rule30_step')


def simulate(n_cells, n_steps, extract_center=False, center_out_path=None):
    """Run Rule 30 simulation on GPU with tqdm progress bar.

    Center column extraction now happens entirely on GPU — no per-step sync.
    """
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

    # GPU-side center column buffer (packed bits)
    if extract_center:
        n_center_bytes = (n_steps + 7) // 8
        # Round up to multiple of 4 for atomicOr alignment
        n_center_bytes_aligned = ((n_center_bytes + 3) // 4) * 4
        center_buf = cp.zeros(n_center_bytes_aligned, dtype=cp.uint8)
        print(f"Center column buffer: {n_center_bytes_aligned:,} bytes on GPU")

    block_size = 256
    grid_size = (n_words + block_size - 1) // block_size

    mempool = cp.get_default_memory_pool()
    cp.cuda.Stream.null.synchronize()
    print(f"Initial center word: {int(tape_a[center_word_idx]):#018x}")

    start = time.perf_counter()
    current = tape_a
    next_buf = tape_b

    # Use batched tqdm updates (every 10000 steps) to minimize overhead
    update_interval = max(1, n_steps // 1000)
    pbar = tqdm(total=n_steps, desc="Rule 30", unit="step", unit_scale=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                miniters=update_interval)

    if extract_center:
        for step in range(n_steps):
            rule30_with_center_kernel(
                (grid_size,), (block_size,),
                (current, next_buf, n_words,
                 center_word_idx, center_bit_idx,
                 center_buf, step)
            )
            current, next_buf = next_buf, current
            if step % update_interval == 0:
                pbar.update(update_interval)
        pbar.update(n_steps - (n_steps // update_interval) * update_interval)
    else:
        for step in range(n_steps):
            rule30_kernel((grid_size,), (block_size,), (current, next_buf, n_words))
            current, next_buf = next_buf, current
            if step % update_interval == 0:
                pbar.update(update_interval)
        pbar.update(n_steps - (n_steps // update_interval) * update_interval)

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
        # Transfer center column from GPU to CPU
        center_bytes = cp.asnumpy(center_buf[:n_center_bytes])
        os.makedirs(os.path.dirname(center_out_path), exist_ok=True)
        with open(center_out_path, 'wb') as f:
            f.write(center_bytes)
        print(f"Center column saved: {center_out_path} ({len(center_bytes):,} bytes = {n_steps:,} bits)")
        results["center_col_file"] = center_out_path
        results["center_col_bits"] = n_steps

        # Verify first 20 bits against CPU reference
        expected = [1,1,0,1,1,1,0,0,1,1,0,0,0,1,0,1,1,0,0,1]
        actual = [(int(center_bytes[i // 8]) >> (i % 8)) & 1 for i in range(min(20, n_steps))]
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

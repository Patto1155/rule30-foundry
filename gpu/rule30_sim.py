"""
GPU-accelerated Rule 30 simulation using CuPy bit-packing.

Rule 30: new[i] = left[i] XOR (center[i] OR right[i])
Stores tape as array of uint64, each holding 64 cells.

Center column extraction stores one byte per step on GPU (simple, no atomics).
"""
import sys
import time
import os
import json
import cupy as cp
import numpy as np
from tqdm import tqdm

# Rule 30 step + center bit extraction
# Thread handling the center word also writes the center bit to output buffer
rule30_with_center_kernel = cp.RawKernel(r'''
extern "C" __global__
void rule30_step_center(
    const unsigned long long* tape,
    unsigned long long* out,
    int n_words,
    int center_word_idx,
    int center_bit_idx,
    unsigned char* center_out,  // one byte per step (0 or 1)
    long long step              // current step number (long long for >2B steps)
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_words) return;

    unsigned long long center = tape[idx];

    // Extract center bit
    if (idx == center_word_idx) {
        center_out[step] = (unsigned char)((center >> center_bit_idx) & 1ULL);
    }

    unsigned long long prev_word = (idx > 0) ? tape[idx - 1] : 0ULL;
    unsigned long long next_word = (idx < n_words - 1) ? tape[idx + 1] : 0ULL;

    unsigned long long left_word = (center << 1) | (prev_word >> 63);
    unsigned long long right_word = (center >> 1) | (next_word << 63);

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

    unsigned long long left_word = (center << 1) | (prev_word >> 63);
    unsigned long long right_word = (center >> 1) | (next_word << 63);

    out[idx] = left_word ^ (center | right_word);
}
''', 'rule30_step')


def _pack_row(row):
    words = np.zeros((len(row) + 63) // 64, dtype=np.uint64)
    for i, bit in enumerate(row):
        if bit:
            words[i // 64] |= np.uint64(1) << np.uint64(i % 64)
    return words


def _unpack_words(words, n_cells):
    row = np.zeros(n_cells, dtype=np.uint8)
    for i in range(n_cells):
        row[i] = (int(words[i // 64]) >> (i % 64)) & 1
    return row


def _naive_step(row):
    out = np.zeros_like(row)
    for i in range(len(row)):
        left = row[i - 1] if i > 0 else 0
        center = row[i]
        right = row[i + 1] if i + 1 < len(row) else 0
        out[i] = left ^ (center | right)
    return out


def verify_gpu_kernel():
    """Check packed GPU kernels against naive Rule 30 across word boundaries."""
    n_words = 3
    n_cells = n_words * 64
    center_word_idx = 1
    center_bit_idx = 0
    row = np.zeros(n_cells, dtype=np.uint8)

    for pos in (63, 64, 65, 95):
        row[pos] = 1

    expected = _naive_step(row)
    tape = cp.asarray(_pack_row(row))
    out = cp.zeros_like(tape)
    out_with_center = cp.zeros_like(tape)
    center_out = cp.zeros(1, dtype=cp.uint8)

    rule30_kernel((1,), (256,), (tape, out, n_words))
    rule30_with_center_kernel(
        (1,), (256,),
        (tape, out_with_center, n_words, center_word_idx, center_bit_idx,
         center_out, np.int64(0))
    )
    cp.cuda.Stream.null.synchronize()

    actual = _unpack_words(cp.asnumpy(out), n_cells)
    actual_with_center = _unpack_words(cp.asnumpy(out_with_center), n_cells)
    if not np.array_equal(actual, expected) or not np.array_equal(actual_with_center, expected):
        diffs = np.where((actual != expected) | (actual_with_center != expected))[0][:10].tolist()
        raise RuntimeError(
            "Packed GPU Rule 30 kernels failed naive word-boundary check; "
            f"first differing cells: {diffs}"
        )

    expected_center = int(row[center_word_idx * 64 + center_bit_idx])
    if int(cp.asnumpy(center_out)[0]) != expected_center:
        raise RuntimeError("GPU center extraction failed pre-step verification.")


def simulate(n_cells, n_steps, extract_center=False, center_out_path=None):
    """Run Rule 30 simulation on GPU with tqdm progress bar."""
    verify_gpu_kernel()

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

    # GPU-side center column buffer: 1 byte per step
    if extract_center:
        center_gpu = cp.zeros(n_steps, dtype=cp.uint8)
        gpu_mem = n_steps / 1024 / 1024
        print(f"Center column buffer: {gpu_mem:.1f} MB on GPU ({n_steps:,} bytes)")

    block_size = 256
    grid_size = (n_words + block_size - 1) // block_size

    mempool = cp.get_default_memory_pool()
    cp.cuda.Stream.null.synchronize()
    print(f"Initial center word: {int(tape_a[center_word_idx]):#018x}")

    start = time.perf_counter()
    current = tape_a
    next_buf = tape_b

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
                 center_gpu, np.int64(step))
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
        # Transfer from GPU, pack bits, save
        print("Transferring center column from GPU...")
        center_bits = cp.asnumpy(center_gpu)  # array of 0/1 bytes
        print(f"  Fraction of 1s: {np.mean(center_bits):.6f} (expect ~0.5)")

        # Pack into bits: bit i of byte j = center_bits[j*8 + i]
        center_packed = np.packbits(center_bits, bitorder='little')

        os.makedirs(os.path.dirname(center_out_path), exist_ok=True)
        with open(center_out_path, 'wb') as f:
            f.write(center_packed)
        print(f"Center column saved: {center_out_path} ({len(center_packed):,} bytes = {n_steps:,} bits)")
        results["center_col_file"] = center_out_path
        results["center_col_bits"] = n_steps

        # Verify first 20 bits against CPU reference
        expected = [1,1,0,1,1,1,0,0,1,1,0,0,0,1,0,1,1,0,0,1]
        actual = center_bits[:20].tolist()
        match = actual == expected[:len(actual)]
        print(f"First 20 bits: {actual}")
        print(f"Expected:      {expected}")
        print(f"Verification:  {'PASS' if match else 'FAIL'}")
        results["verification_passed"] = match
        results["fraction_ones"] = float(np.mean(center_bits))

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

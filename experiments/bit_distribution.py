#!/usr/bin/env python
"""Experiment A — Bit Distribution Analysis of Rule 30 Center Column.

Counts 0s and 1s over 10M bits, tracks running bias at 100k-step checkpoints,
and tests whether bias converges as 1/sqrt(N) (consistent with fair coin).

Directly relevant to Wolfram Problem 3 (statistical properties).
"""

import sys
import os
import csv
import time
import datetime
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE = r"D:\APATPROJECTS\rule30-research\data\center_col_10M.bin"
OUT_CSV = r"D:\APATPROJECTS\rule30-research\data\bit_distribution.csv"
LOG_FILE = r"D:\APATPROJECTS\rule30-research\docs\experiment-logs\A_bit_distribution.md"
TOTAL_BITS = 10_000_000
CHECKPOINT = 100_000  # record bias every 100k steps

# ---------------------------------------------------------------------------
# CuPy / NumPy portability
# ---------------------------------------------------------------------------
try:
    import cupy as xp
    GPU = True
except ImportError:
    xp = np
    GPU = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_center_column(path: str, n_bits: int) -> np.ndarray:
    """Load packed-bit file and return a 1-D uint8 array of individual bits."""
    n_bytes = (n_bits + 7) // 8
    raw = np.fromfile(path, dtype=np.uint8, count=n_bytes)
    # Unpack: bit 0 of byte 0 = step 0, bit 1 of byte 0 = step 1, etc.
    bits = np.unpackbits(raw, bitorder='little')[:n_bits]
    return bits


def write_log(results: dict):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"""# Experiment Log

- Date: {datetime.date.today().isoformat()}
- Title: Bit Distribution of Rule 30 Center Column (10M steps)
- Goal: Measure the 0/1 bias of the center column and check 1/sqrt(N) convergence
- Setup: 10M-bit center column from {DATA_FILE}, NumPy {'+ CuPy (GPU)' if GPU else '(CPU only)'}
- Method: Count 0s and 1s cumulatively at every {CHECKPOINT}-step checkpoint; compute bias = (count_1 - count_0) / total; fit expected 1/sqrt(N) envelope
- Result: Final bias = {results['final_bias']:.8f} at N = {results['total']}; expected |bias| ~ 1/sqrt(N) = {results['expected_bound']:.8f}; ratio |bias|/bound = {results['ratio']:.4f}
- Interpretation: {'Bias is within expected 1/sqrt(N) bound — consistent with unbiased coin.' if results['ratio'] <= 3.0 else 'Bias EXCEEDS 3x the 1/sqrt(N) bound — potential systematic bias detected!'}
- Next Step: Run block-frequency and autocorrelation experiments for deeper structure tests
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.isfile(DATA_FILE):
        print(f"ERROR: Data file not found: {DATA_FILE}")
        print("Generate center_col_10M.bin first, then re-run this script.")
        sys.exit(1)

    print("=" * 60)
    print("Experiment A — Bit Distribution (Rule 30 Center Column)")
    print("=" * 60)
    print(f"Data file : {DATA_FILE}")
    print(f"Total bits: {TOTAL_BITS:,}")
    print(f"GPU       : {'Yes (CuPy)' if GPU else 'No (NumPy)'}")
    print()

    t0 = time.perf_counter()

    # Load data
    print("Loading center column data...")
    bits = load_center_column(DATA_FILE, TOTAL_BITS)
    print(f"  Loaded {len(bits):,} bits in {time.perf_counter() - t0:.2f}s")

    # Running bias at checkpoints
    n_checkpoints = TOTAL_BITS // CHECKPOINT
    rows = []
    count_1 = 0
    count_0 = 0

    print(f"\nComputing running bias at {n_checkpoints} checkpoints...")
    for i in tqdm(range(n_checkpoints), desc="Checkpoints"):
        start = i * CHECKPOINT
        end = start + CHECKPOINT
        chunk = bits[start:end]
        ones = int(np.sum(chunk))
        zeros = CHECKPOINT - ones
        count_1 += ones
        count_0 += zeros
        total = count_1 + count_0
        bias = (count_1 - count_0) / total
        rows.append((total, count_0, count_1, bias))

    # Save CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "count_0", "count_1", "bias"])
        for row in rows:
            writer.writerow(row)
    print(f"Saved bias convergence data to {OUT_CSV}")

    # Final statistics
    final_bias = rows[-1][3]
    total = rows[-1][0]
    expected_bound = 1.0 / np.sqrt(total)
    ratio = abs(final_bias) / expected_bound if expected_bound > 0 else float('inf')

    results = {
        'final_bias': final_bias,
        'total': total,
        'count_0': count_0,
        'count_1': count_1,
        'expected_bound': expected_bound,
        'ratio': ratio,
    }

    # Print summary
    elapsed = time.perf_counter() - t0
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Total bits analysed : {total:,}")
    print(f"  Count 0             : {count_0:,}")
    print(f"  Count 1             : {count_1:,}")
    print(f"  Final bias          : {final_bias:+.8f}")
    print(f"  Expected |bias| (1/sqrt(N)): {expected_bound:.8f}")
    print(f"  |bias| / bound      : {ratio:.4f}")
    if ratio <= 3.0:
        print("  VERDICT: Bias within expected 1/sqrt(N) envelope (fair coin).")
    else:
        print("  VERDICT: Bias EXCEEDS 3x expected bound — possible systematic bias!")
    print(f"  Elapsed time        : {elapsed:.2f}s")
    print()

    # Write experiment log
    write_log(results)
    print(f"Experiment log saved to {LOG_FILE}")


if __name__ == "__main__":
    main()

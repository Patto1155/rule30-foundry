#!/usr/bin/env python
"""Experiment C — Block Frequency Analysis of Rule 30 Center Column.

For block sizes k=1..20, counts all 2^k possible bit patterns in the
center column, computes chi-squared deviation from uniform distribution,
and records the most over/under-represented patterns.
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
DATA_DIR = r"D:\APATPROJECTS\rule30-research\data"
SUMMARY_CSV = os.path.join(DATA_DIR, "block_frequency_summary.csv")
LOG_FILE = r"D:\APATPROJECTS\rule30-research\docs\experiment-logs\C_block_frequency.md"
TOTAL_BITS = 10_000_000
K_MIN = 1
K_MAX = 20

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
    """Load packed-bit file and return 1-D uint8 array of individual bits."""
    n_bytes = (n_bits + 7) // 8
    raw = np.fromfile(path, dtype=np.uint8, count=n_bytes)
    bits = np.unpackbits(raw, bitorder='little')[:n_bits]
    return bits


def count_blocks(bits: np.ndarray, k: int) -> np.ndarray:
    """Count occurrences of each k-bit pattern (overlapping windows).

    Returns array of length 2^k with counts for patterns 0..2^k-1.
    Pattern value: bits[i]*2^0 + bits[i+1]*2^1 + ... + bits[i+k-1]*2^(k-1).
    """
    n = len(bits) - k + 1
    n_patterns = 1 << k
    counts = np.zeros(n_patterns, dtype=np.int64)

    # Build pattern indices using sliding window dot product
    # For k <= 20 this is efficient with vectorised operations
    powers = (1 << np.arange(k, dtype=np.int64))  # [1, 2, 4, ..., 2^(k-1)]

    # Process in chunks to avoid memory issues
    chunk_size = min(n, 2_000_000)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        # Build a (end-start, k) matrix of windows
        indices = np.lib.stride_tricks.sliding_window_view(bits[start:start + (end - start) + k - 1], k)
        pattern_vals = indices @ powers
        # Count via bincount
        chunk_counts = np.bincount(pattern_vals, minlength=n_patterns)
        counts += chunk_counts

    return counts


def chi_squared_test(counts: np.ndarray, expected_per_bin: float):
    """Compute chi-squared statistic and approximate p-value."""
    from scipy import stats
    chi2 = np.sum((counts - expected_per_bin) ** 2 / expected_per_bin)
    df = len(counts) - 1
    p_value = 1.0 - stats.chi2.cdf(chi2, df)
    return float(chi2), float(p_value), int(df)


def main():
    if not os.path.isfile(DATA_FILE):
        print(f"ERROR: Data file not found: {DATA_FILE}")
        print("Generate center_col_10M.bin first, then re-run this script.")
        sys.exit(1)

    print("=" * 60)
    print("Experiment C — Block Frequency Analysis (Rule 30 Center Column)")
    print("=" * 60)
    print(f"Data file  : {DATA_FILE}")
    print(f"Total bits : {TOTAL_BITS:,}")
    print(f"Block sizes: k={K_MIN}..{K_MAX}")
    print()

    t0 = time.perf_counter()

    # Check for scipy
    try:
        from scipy import stats
    except ImportError:
        print("ERROR: scipy is required for chi-squared p-values.")
        print("Install with: C:\\Python313\\python.exe -m pip install scipy")
        sys.exit(1)

    # Load data
    print("Loading center column data...")
    bits = load_center_column(DATA_FILE, TOTAL_BITS)
    print(f"  Loaded {len(bits):,} bits in {time.perf_counter() - t0:.2f}s")

    # Analyse each block size
    summary_rows = []
    os.makedirs(DATA_DIR, exist_ok=True)

    print()
    for k in tqdm(range(K_MIN, K_MAX + 1), desc="Block sizes"):
        n_windows = TOTAL_BITS - k + 1
        n_patterns = 1 << k
        expected = n_windows / n_patterns

        counts = count_blocks(bits, k)
        chi2, p_value, df = chi_squared_test(counts, expected)

        # Most over-represented pattern
        max_idx = int(np.argmax(counts))
        max_count = int(counts[max_idx])
        max_pattern = format(max_idx, f'0{k}b')[::-1]  # reverse to show bit order

        # Most under-represented pattern
        min_idx = int(np.argmin(counts))
        min_count = int(counts[min_idx])
        min_pattern = format(min_idx, f'0{k}b')[::-1]

        summary_rows.append({
            'k': k,
            'n_patterns': n_patterns,
            'n_windows': n_windows,
            'expected_per_pattern': expected,
            'chi2': chi2,
            'df': df,
            'p_value': p_value,
            'max_pattern': max_pattern,
            'max_count': max_count,
            'max_ratio': max_count / expected,
            'min_pattern': min_pattern,
            'min_count': min_count,
            'min_ratio': min_count / expected,
        })

        # Save per-k frequency table
        freq_csv = os.path.join(DATA_DIR, f"block_freq_k{k}.csv")
        with open(freq_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["pattern_value", "pattern_bits", "count", "expected", "ratio"])
            for idx in range(n_patterns):
                pat_str = format(idx, f'0{k}b')[::-1]
                writer.writerow([idx, pat_str, int(counts[idx]),
                                 f"{expected:.2f}", f"{counts[idx]/expected:.6f}"])

    # Save summary
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "n_patterns", "n_windows", "expected_per_pattern",
                          "chi2", "df", "p_value",
                          "max_pattern", "max_count", "max_ratio",
                          "min_pattern", "min_count", "min_ratio"])
        for row in summary_rows:
            writer.writerow([
                row['k'], row['n_patterns'], row['n_windows'],
                f"{row['expected_per_pattern']:.2f}",
                f"{row['chi2']:.4f}", row['df'], f"{row['p_value']:.6e}",
                row['max_pattern'], row['max_count'], f"{row['max_ratio']:.6f}",
                row['min_pattern'], row['min_count'], f"{row['min_ratio']:.6f}",
            ])
    print(f"\nSaved summary to {SUMMARY_CSV}")

    # Print results
    elapsed = time.perf_counter() - t0
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'k':>2}  {'chi2':>12}  {'p-value':>12}  {'max pattern':>12}  {'max ratio':>10}  {'verdict':>10}")
    print(f"  {'—'*2}  {'—'*12}  {'—'*12}  {'—'*12}  {'—'*10}  {'—'*10}")
    n_significant = 0
    for row in summary_rows:
        verdict = "FAIL" if row['p_value'] < 0.01 else "pass"
        if row['p_value'] < 0.01:
            n_significant += 1
        print(f"  {row['k']:>2}  {row['chi2']:>12.2f}  {row['p_value']:>12.4e}  "
              f"{row['max_pattern']:>12}  {row['max_ratio']:>10.4f}  {verdict:>10}")
    print()
    if n_significant == 0:
        print("  VERDICT: All block sizes pass chi-squared test (p >= 0.01).")
        print("           Block frequencies are consistent with uniform distribution.")
    else:
        print(f"  VERDICT: {n_significant}/{K_MAX} block sizes show significant deviation (p < 0.01).")
        print("           Non-uniform structure detected at these scales!")
    print(f"  Elapsed time: {elapsed:.2f}s")
    print()

    # Write experiment log
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    sig_ks = [r['k'] for r in summary_rows if r['p_value'] < 0.01]
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"""# Experiment Log

- Date: {datetime.date.today().isoformat()}
- Title: Block Frequency Analysis of Rule 30 Center Column (k=1..{K_MAX})
- Goal: Test whether k-bit block frequencies match uniform distribution for each k
- Setup: 10M-bit center column; overlapping k-bit windows; chi-squared goodness-of-fit test
- Method: For each k=1..{K_MAX}, count all 2^k patterns in overlapping windows; compute chi-squared vs uniform expectation; flag p < 0.01
- Result: {n_significant}/{K_MAX} block sizes show significant deviation (p < 0.01). Significant k values: {sig_ks if sig_ks else 'none'}
- Interpretation: {'All block sizes consistent with uniform — strong evidence for equidistribution.' if n_significant == 0 else f'Deviations at k={sig_ks} — may indicate short-range structure or insufficient data for large k.'}
- Next Step: Cross-reference any deviations with autocorrelation results; proceed to Markov predictor experiment
""")
    print(f"Experiment log saved to {LOG_FILE}")


if __name__ == "__main__":
    main()

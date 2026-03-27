#!/usr/bin/env python
"""Experiment E — Period Search in Rule 30 Center Column.

Searches for any repeating period in the center column using a sampling-based
approach (Rabin-Karp style rolling comparison).  For each candidate period p
from 1 to 1,000,000, estimates the match rate center_col[i] == center_col[i+p]
over a large sample.  A perfect period would give match rate = 1.0.

This is a direct attack on Wolfram Problem 1 (does the center column repeat?).
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
OUT_CSV = r"D:\APATPROJECTS\rule30-research\data\period_search.csv"
LOG_FILE = r"D:\APATPROJECTS\rule30-research\docs\experiment-logs\E_period_search.md"
TOTAL_BITS = 10_000_000
MAX_PERIOD = 1_000_000
# Number of sample positions to test per candidate period
SAMPLE_SIZE = 10_000
# Periods to test exhaustively (all positions) if they look promising
EXHAUSTIVE_THRESHOLD = 0.55  # match rate above this triggers exhaustive check
# Save top-N results
TOP_N = 100

# ---------------------------------------------------------------------------
# CuPy / NumPy portability
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    GPU = True
except ImportError:
    cp = None
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


def test_period_sampled(bits: np.ndarray, p: int, n_samples: int, rng: np.random.Generator) -> float:
    """Test period p by sampling random positions and checking agreement."""
    max_start = len(bits) - p
    if max_start <= 0:
        return 0.5
    indices = rng.integers(0, max_start, size=n_samples)
    matches = np.sum(bits[indices] == bits[indices + p])
    return float(matches) / n_samples


def test_period_exhaustive(bits: np.ndarray, p: int) -> float:
    """Test period p exhaustively: check all valid positions."""
    n = len(bits) - p
    matches = np.sum(bits[:n] == bits[p:p + n])
    return float(matches) / n


def main():
    if not os.path.isfile(DATA_FILE):
        print(f"ERROR: Data file not found: {DATA_FILE}")
        print("Generate center_col_10M.bin first, then re-run this script.")
        sys.exit(1)

    print("=" * 60)
    print("Experiment E — Period Search (Rule 30 Center Column)")
    print("=" * 60)
    print(f"Data file   : {DATA_FILE}")
    print(f"Total bits  : {TOTAL_BITS:,}")
    print(f"Max period  : {MAX_PERIOD:,}")
    print(f"Sample size : {SAMPLE_SIZE:,} per period")
    print(f"GPU         : {'Yes (CuPy)' if GPU else 'No (NumPy)'}")
    print()

    t0 = time.perf_counter()

    # Load data
    print("Loading center column data...")
    bits = load_center_column(DATA_FILE, TOTAL_BITS)
    print(f"  Loaded {len(bits):,} bits in {time.perf_counter() - t0:.2f}s")

    # Phase 1: Sampled scan of all periods
    rng = np.random.default_rng(seed=42)
    print(f"\nPhase 1: Scanning periods 1..{MAX_PERIOD:,} (sampled, {SAMPLE_SIZE:,} positions each)...")

    match_rates = np.zeros(MAX_PERIOD, dtype=np.float64)

    # Process in batches for efficiency with GPU if available
    if GPU and cp is not None:
        print("  Using GPU-accelerated comparison...")
        bits_gpu = cp.asarray(bits)
        batch_size = 1000
        for batch_start in tqdm(range(0, MAX_PERIOD, batch_size), desc="Period scan"):
            batch_end = min(batch_start + batch_size, MAX_PERIOD)
            for p_idx in range(batch_start, batch_end):
                p = p_idx + 1
                max_start = len(bits) - p
                if max_start <= 0:
                    match_rates[p_idx] = 0.5
                    continue
                indices = rng.integers(0, max_start, size=SAMPLE_SIZE)
                indices_gpu = cp.asarray(indices)
                matches = int(cp.sum(bits_gpu[indices_gpu] == bits_gpu[indices_gpu + p]))
                match_rates[p_idx] = matches / SAMPLE_SIZE
    else:
        for p in tqdm(range(1, MAX_PERIOD + 1), desc="Period scan"):
            match_rates[p - 1] = test_period_sampled(bits, p, SAMPLE_SIZE, rng)

    t_phase1 = time.perf_counter() - t0

    # Expected match rate for random sequence: 0.5
    # Standard error of sampled match rate: sqrt(0.25 / SAMPLE_SIZE) ~ 0.005
    se = np.sqrt(0.25 / SAMPLE_SIZE)
    z_scores = (match_rates - 0.5) / se

    # Find promising periods
    promising = np.where(match_rates > EXHAUSTIVE_THRESHOLD)[0]
    print(f"\n  Phase 1 done in {t_phase1:.1f}s")
    print(f"  Periods with match rate > {EXHAUSTIVE_THRESHOLD}: {len(promising)}")

    # Phase 2: Exhaustive check on promising periods
    exhaustive_results = {}
    if len(promising) > 0:
        print(f"\nPhase 2: Exhaustive check on {len(promising)} promising periods...")
        for p_idx in tqdm(promising, desc="Exhaustive check"):
            p = int(p_idx + 1)
            rate = test_period_exhaustive(bits, p)
            exhaustive_results[p] = rate
            match_rates[p_idx] = rate  # update with exact value

    # Sort all periods by match rate, take top N
    top_indices = np.argsort(match_rates)[-TOP_N:][::-1]

    # Save results
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "period", "match_rate", "z_score", "exhaustive_checked"])
        for rank, idx in enumerate(top_indices, 1):
            p = int(idx + 1)
            writer.writerow([
                rank, p,
                f"{match_rates[idx]:.8f}",
                f"{z_scores[idx]:.4f}",
                "yes" if p in exhaustive_results else "no",
            ])
    print(f"\nSaved top-{TOP_N} results to {OUT_CSV}")

    # Results summary
    best_idx = top_indices[0]
    best_period = int(best_idx + 1)
    best_rate = match_rates[best_idx]
    best_z = z_scores[best_idx]

    elapsed = time.perf_counter() - t0
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Expected match rate (random): 0.5000")
    print(f"  Sampling SE                 : {se:.6f}")
    print(f"  Best period found           : {best_period}")
    print(f"  Best match rate             : {best_rate:.8f}")
    print(f"  Best z-score                : {best_z:.4f}")
    print()
    print(f"  Top-10 periods by match rate:")
    for rank, idx in enumerate(top_indices[:10], 1):
        p = int(idx + 1)
        exh = " (exhaustive)" if p in exhaustive_results else ""
        print(f"    {rank:>2}. period={p:>7d}  match_rate={match_rates[idx]:.6f}  z={z_scores[idx]:+.2f}{exh}")
    print()

    if best_rate > 0.99:
        verdict = "STRONG PERIOD CANDIDATE FOUND! Match rate > 99% — may be a true period."
    elif best_rate > 0.55:
        verdict = "Weak periodic signal detected — warrants further investigation with more data."
    else:
        verdict = "No significant period found. Center column appears aperiodic up to period 1M."
    print(f"  VERDICT: {verdict}")
    print(f"  Elapsed time: {elapsed:.2f}s")
    print()

    # Write experiment log
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"""# Experiment Log

- Date: {datetime.date.today().isoformat()}
- Title: Period Search in Rule 30 Center Column (periods 1..{MAX_PERIOD:,})
- Goal: Search for any repeating period in the center column — a direct attack on Wolfram Problem 1
- Setup: 10M-bit center column; sampled match test ({SAMPLE_SIZE:,} positions per period); exhaustive verification for promising candidates
- Method: For each candidate period p, estimate P(bit[i] == bit[i+p]) via sampling; flag periods where match rate > {EXHAUSTIVE_THRESHOLD}; verify flagged periods exhaustively
- Result: Best period = {best_period} with match rate {best_rate:.8f} (z={best_z:.2f}). {len(promising)} periods exceeded threshold for exhaustive checking.
- Interpretation: {verdict}
- Next Step: {'Verify candidate period with independent data generation; test multiples/divisors of candidate period' if best_rate > 0.55 else 'Extend search to larger periods using FFT-based methods; consider quasi-periodic analysis'}
""")
    print(f"Experiment log saved to {LOG_FILE}")


if __name__ == "__main__":
    main()

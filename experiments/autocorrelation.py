#!/usr/bin/env python
"""Experiment B — Autocorrelation Scan of Rule 30 Center Column.

Computes autocorrelation at lags 1..100,000 via FFT.  Uses CuPy for GPU-
accelerated FFT when available.  Any significant non-zero autocorrelation
at any lag would be a major finding.
"""

import sys
import os
import time
import datetime
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE = r"D:\APATPROJECTS\rule30-research\data\center_col_10M.bin"
OUT_NPY = r"D:\APATPROJECTS\rule30-research\data\autocorrelation.npy"
OUT_CSV = r"D:\APATPROJECTS\rule30-research\data\autocorrelation_top20.csv"
LOG_FILE = r"D:\APATPROJECTS\rule30-research\docs\experiment-logs\B_autocorrelation.md"
TOTAL_BITS = 10_000_000
MAX_LAG = 100_000

# ---------------------------------------------------------------------------
# CuPy / NumPy portability
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    # Verify cuFFT is available (needed for FFT operations)
    from cupy.cuda import cufft  # noqa: F401
    GPU = True
except (ImportError, Exception):
    cp = None
    GPU = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_center_column(path: str, n_bits: int) -> np.ndarray:
    """Load packed-bit file and return 1-D float64 array of {-1, +1}."""
    n_bytes = (n_bits + 7) // 8
    raw = np.fromfile(path, dtype=np.uint8, count=n_bytes)
    bits = np.unpackbits(raw, bitorder='little')[:n_bits]
    # Map 0->-1, 1->+1 for zero-mean autocorrelation
    return bits.astype(np.float64) * 2.0 - 1.0


def autocorrelation_fft(signal, max_lag, use_gpu=False):
    """Compute normalised autocorrelation via FFT for lags 0..max_lag."""
    n = len(signal)
    # Pad to next power of 2 for efficient FFT
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2

    if use_gpu and cp is not None:
        x = cp.asarray(signal)
        X = cp.fft.rfft(x, n=fft_size)
        power = X * cp.conj(X)
        acorr_full = cp.fft.irfft(power, n=fft_size)[:max_lag + 1]
        acorr = cp.asnumpy(acorr_full)
    else:
        X = np.fft.rfft(signal, n=fft_size)
        power = X * np.conj(X)
        acorr_full = np.fft.irfft(power, n=fft_size)[:max_lag + 1]
        acorr = acorr_full.copy()

    # Normalise by lag-0 (variance)
    acorr /= acorr[0]
    return acorr


def write_log(top20, acorr, elapsed):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    top_lines = "\n".join(
        f"    lag={lag:>7d}  r={acorr[lag]:+.8f}"
        for lag, _ in top20
    )
    max_abs = max(abs(v) for _, v in top20)
    threshold = 2.0 / np.sqrt(TOTAL_BITS)  # ~2-sigma for white noise

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"""# Experiment Log

- Date: {datetime.date.today().isoformat()}
- Title: Autocorrelation Scan of Rule 30 Center Column (lags 1..{MAX_LAG:,})
- Goal: Detect any linear dependence between the center column and lagged copies of itself
- Setup: 10M-bit center column mapped to +/-1; FFT-based autocorrelation; {'CuPy GPU' if GPU else 'NumPy CPU'}
- Method: Compute normalised autocorrelation r(lag) = E[x_t * x_(t+lag)] / Var(x) via FFT for lags 1..{MAX_LAG:,}; rank by |r|
- Result: Top-20 lags by |autocorrelation|:
{top_lines}
    Max |r| = {max_abs:.8f}; 2-sigma noise floor = {threshold:.8f}
- Interpretation: {'All autocorrelations are within noise floor — no detectable linear structure.' if max_abs < 3 * threshold else 'Some autocorrelations EXCEED noise floor — potential linear structure detected!'}
- Next Step: If any lag is significant, investigate block patterns around that lag; otherwise proceed to block frequency analysis
""")


def main():
    if not os.path.isfile(DATA_FILE):
        print(f"ERROR: Data file not found: {DATA_FILE}")
        print("Generate center_col_10M.bin first, then re-run this script.")
        sys.exit(1)

    print("=" * 60)
    print("Experiment B — Autocorrelation Scan (Rule 30 Center Column)")
    print("=" * 60)
    print(f"Data file : {DATA_FILE}")
    print(f"Total bits: {TOTAL_BITS:,}")
    print(f"Max lag   : {MAX_LAG:,}")
    print(f"GPU       : {'Yes (CuPy)' if GPU else 'No (NumPy)'}")
    print()

    t0 = time.perf_counter()

    # Load data
    print("Loading center column data...")
    signal = load_center_column(DATA_FILE, TOTAL_BITS)
    print(f"  Loaded {len(signal):,} values in {time.perf_counter() - t0:.2f}s")

    # Compute autocorrelation
    print(f"\nComputing autocorrelation via FFT (lags 0..{MAX_LAG:,})...")
    acorr = autocorrelation_fft(signal, MAX_LAG, use_gpu=GPU)
    t_fft = time.perf_counter() - t0
    print(f"  FFT autocorrelation done in {t_fft:.2f}s")

    # Save full array
    os.makedirs(os.path.dirname(OUT_NPY), exist_ok=True)
    np.save(OUT_NPY, acorr)
    print(f"  Saved full autocorrelation ({len(acorr):,} lags) to {OUT_NPY}")

    # Find top-20 by absolute value (exclude lag 0)
    abs_acorr = np.abs(acorr[1:])  # lags 1..MAX_LAG
    top20_indices = np.argsort(abs_acorr)[-20:][::-1]  # descending
    top20 = [(int(idx + 1), float(acorr[idx + 1])) for idx in top20_indices]

    # Save top-20 CSV
    import csv
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "lag", "autocorrelation", "abs_autocorrelation"])
        for rank, (lag, val) in enumerate(top20, 1):
            writer.writerow([rank, lag, f"{val:.10f}", f"{abs(val):.10f}"])
    print(f"  Saved top-20 lags to {OUT_CSV}")

    # Statistics
    noise_floor = 2.0 / np.sqrt(TOTAL_BITS)  # 2-sigma for white noise
    max_abs = max(abs(v) for _, v in top20)

    elapsed = time.perf_counter() - t0
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Noise floor (2-sigma) : {noise_floor:.8f}")
    print(f"  Max |autocorrelation| : {max_abs:.8f}  (lag {top20[0][0]})")
    print()
    print("  Top-20 lags by |autocorrelation|:")
    for rank, (lag, val) in enumerate(top20, 1):
        flag = " ***" if abs(val) > 3 * noise_floor else ""
        print(f"    {rank:>2}. lag={lag:>7d}  r={val:+.8f}{flag}")
    print()
    if max_abs < 3 * noise_floor:
        print("  VERDICT: No autocorrelation exceeds 3x noise floor.")
        print("           Center column appears linearly uncorrelated.")
    else:
        print("  VERDICT: Some lags exceed 3x noise floor!")
        print("           Potential linear structure detected — investigate further.")
    print(f"  Elapsed time: {elapsed:.2f}s")
    print()

    write_log(top20, acorr, elapsed)
    print(f"Experiment log saved to {LOG_FILE}")


if __name__ == "__main__":
    main()

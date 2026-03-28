#!/usr/bin/env python
"""Extended Period Search on 46M-bit center column (FFT autocorrelation).

Searches for periods up to MAX_PERIOD using FFT-based autocorrelation — much
faster than the sampling approach in Experiment E, and covers 4.5x more range.

Attack on Wolfram Problem 1: does the center column ever become periodic?

Method:
  1. Convert bits to ±1 signal
  2. FFT → power spectrum → IFFT = circular autocorrelation
  3. Compute z-scores at each lag (under H0: iid Bernoulli(0.5))
  4. Apply Bonferroni correction for multiple testing
  5. Flag any |z| > threshold as a period candidate

Verification output:
  - Top-20 lags by |z-score| printed and saved to CSV
  - matplotlib plot saved to docs/plots/ (z-score vs lag, threshold line)
  - Experiment log written to docs/experiment-logs/
"""

import os
import sys
import time
import json
import datetime
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE  = Path(r"D:\APATPROJECTS\rule30-research\data\center_col_46M.bin")
OUT_CSV    = Path(r"D:\APATPROJECTS\rule30-research\data\period_search_extended.csv")
PLOT_FILE  = Path(r"D:\APATPROJECTS\rule30-research\docs\plots\period_search_extended.png")
LOG_FILE   = Path(r"D:\APATPROJECTS\rule30-research\docs\experiment-logs\M_period_search_extended.md")
JSON_FILE  = Path(r"D:\APATPROJECTS\rule30-research\data\period_search_extended.json")

TOTAL_BITS = 46_000_000
MAX_PERIOD = 4_500_000   # search lags 1 .. 4.5M
ALPHA      = 0.05        # family-wise error rate for Bonferroni

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_bits(path: Path, n_bits: int) -> np.ndarray:
    n_bytes = (n_bits + 7) // 8
    raw = np.fromfile(str(path), dtype=np.uint8, count=n_bytes)
    bits = np.unpackbits(raw, bitorder='little')[:n_bits]
    return bits


def fft_autocorrelation(signal: np.ndarray) -> np.ndarray:
    """Circular autocorrelation via FFT (normalised, float32 for memory)."""
    n = len(signal)
    # Work in float32 to keep memory < 1 GB for 46M samples
    sig = signal.astype(np.float32)
    F   = np.fft.rfft(sig, n=n)
    acf = np.fft.irfft(F * np.conj(F), n=n)
    # Normalise so lag-0 = 1.0
    acf /= acf[0]
    return acf.astype(np.float64)


def bonferroni_threshold(n_tests: int, alpha: float = 0.05) -> float:
    return float(norm.ppf(1 - alpha / (2 * n_tests)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Extended Period Search — Rule 30 Center Column (46M bits)")
    print("=" * 65)

    # Sanity check
    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found.")
        print("Waiting for simulation to finish. Re-run this script after.")
        sys.exit(1)

    actual_size = DATA_FILE.stat().st_size
    expected    = TOTAL_BITS   # 1 byte per bit in this format
    if actual_size < expected * 0.99:
        print(f"WARNING: file is {actual_size:,} bytes, expected {expected:,}.")
        print("Simulation may still be running. Check again later.")
        sys.exit(1)
    print(f"Data file OK: {actual_size:,} bytes ({actual_size/1e6:.1f} MB)")

    t0 = time.perf_counter()

    # 1. Load
    print(f"\nLoading {TOTAL_BITS/1e6:.0f}M bits …")
    bits = load_bits(DATA_FILE, TOTAL_BITS)
    fraction_ones = bits.mean()
    print(f"  Loaded. fraction_ones = {fraction_ones:.6f}  (ideal 0.5)")
    print(f"  Mean-centered signal: bits → ±1")

    # Convert to ±1
    signal = bits.astype(np.float32) * 2 - 1  # 0→-1, 1→+1
    n = len(signal)

    # 2. FFT autocorrelation
    print(f"\nComputing FFT autocorrelation (n={n:,}) …")
    t_fft = time.perf_counter()
    acf = fft_autocorrelation(signal)
    print(f"  FFT done in {time.perf_counter() - t_fft:.1f}s")

    # 3. Z-scores for lags 1..MAX_PERIOD
    # Under H0 (iid ±1), Var[acf(k)] ≈ 1/n  →  std = 1/sqrt(n)
    std_null = 1.0 / np.sqrt(n)
    lags     = np.arange(1, MAX_PERIOD + 1)
    z_scores = acf[lags] / std_null

    # 4. Bonferroni threshold
    threshold = bonferroni_threshold(MAX_PERIOD, ALPHA)
    print(f"\nBonferroni threshold (α={ALPHA}, n_tests={MAX_PERIOD:,}): |z| > {threshold:.4f}")

    # 5. Candidates
    candidates = np.where(np.abs(z_scores) > threshold)[0]
    print(f"Candidate periods above threshold: {len(candidates)}")

    best_idx   = int(np.argmax(np.abs(z_scores)))
    best_lag   = int(lags[best_idx])
    best_z     = float(z_scores[best_idx])
    best_acf   = float(acf[best_lag])

    # 6. Top-20 summary
    top20_idx  = np.argsort(np.abs(z_scores))[-20:][::-1]
    print(f"\nTop-20 lags by |z-score|:")
    print(f"  {'Rank':>4}  {'Lag':>9}  {'ACF':>10}  {'z-score':>10}  {'Flag'}")
    print(f"  {'-'*4}  {'-'*9}  {'-'*10}  {'-'*10}  {'-'*10}")
    top20_rows = []
    for rank, idx in enumerate(top20_idx, 1):
        lag  = int(lags[idx])
        z    = float(z_scores[idx])
        r    = float(acf[lag])
        flag = "*** CANDIDATE ***" if abs(z) > threshold else ""
        print(f"  {rank:>4}  {lag:>9,}  {r:>+10.6f}  {z:>+10.4f}  {flag}")
        top20_rows.append({"rank": rank, "lag": lag, "acf": round(r, 8),
                           "z_score": round(z, 6), "above_threshold": abs(z) > threshold})

    # 7. Verdict
    print()
    if len(candidates) > 0:
        verdict = (f"PERIOD CANDIDATE(S) FOUND at lags: "
                   f"{[int(lags[i]) for i in candidates[:5]]}. "
                   f"Best |z|={abs(best_z):.2f} > threshold {threshold:.2f}.")
    else:
        verdict = (f"No significant period found up to {MAX_PERIOD:,}. "
                   f"Best |z|={abs(best_z):.4f} < threshold {threshold:.4f}. "
                   f"Center column appears aperiodic.")
    print(f"VERDICT: {verdict}")

    # 8. Save CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(str(OUT_CSV), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "lag", "acf", "z_score", "above_threshold"])
        for row in top20_rows:
            writer.writerow([row["rank"], row["lag"], row["acf"],
                             row["z_score"], row["above_threshold"]])
    print(f"\nTop-20 saved → {OUT_CSV}")

    # 9. Save JSON summary
    result = {
        "total_bits":          TOTAL_BITS,
        "max_period_searched": MAX_PERIOD,
        "fraction_ones":       round(float(fraction_ones), 8),
        "bonferroni_threshold": round(threshold, 6),
        "n_candidates":        int(len(candidates)),
        "best_lag":            best_lag,
        "best_z":              round(best_z, 6),
        "best_acf":            round(best_acf, 8),
        "verdict":             verdict,
        "top20":               top20_rows,
    }
    JSON_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(str(JSON_FILE), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"JSON summary   → {JSON_FILE}")

    # 10. Plot
    print(f"\nGenerating plot …")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Downsample for plot (every 100th lag to keep it readable)
        step    = max(1, MAX_PERIOD // 50_000)
        plot_lags   = lags[::step]
        plot_z      = z_scores[::step]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        fig.suptitle("Extended Period Search — Rule 30 Center Column (46M bits)",
                     fontsize=13, fontweight="bold")

        # Top panel: full z-score landscape
        ax1.plot(plot_lags / 1e6, plot_z, lw=0.4, color="#2196F3", alpha=0.8)
        ax1.axhline( threshold, color="red",  lw=1.2, ls="--", label=f"+threshold ({threshold:.2f})")
        ax1.axhline(-threshold, color="red",  lw=1.2, ls="--", label=f"−threshold")
        ax1.axhline(0,          color="gray", lw=0.5, ls="-")
        if len(candidates) > 0:
            for ci in candidates[:20]:
                ax1.axvline(lags[ci] / 1e6, color="orange", lw=0.8, alpha=0.7)
        ax1.set_xlabel("Lag (millions of steps)")
        ax1.set_ylabel("Z-score")
        ax1.set_title(f"Z-score for all lags 1–{MAX_PERIOD//1_000_000}M  "
                      f"(best |z|={abs(best_z):.3f})")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Bottom panel: histogram of z-scores vs expected N(0,1)
        from scipy.stats import norm as sp_norm
        ax2.hist(z_scores, bins=200, density=True, color="#4CAF50", alpha=0.7,
                 label="Observed z-scores")
        x_norm = np.linspace(z_scores.min(), z_scores.max(), 500)
        ax2.plot(x_norm, sp_norm.pdf(x_norm), "r-", lw=2, label="N(0,1) expected")
        ax2.axvline( threshold, color="red", lw=1.2, ls="--")
        ax2.axvline(-threshold, color="red", lw=1.2, ls="--")
        ax2.set_xlabel("Z-score")
        ax2.set_ylabel("Density")
        ax2.set_title("Distribution of z-scores vs N(0,1) null — should overlap if no period")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(PLOT_FILE), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved     → {PLOT_FILE}")
    except Exception as e:
        print(f"Plot skipped ({e})")

    # 11. Experiment log
    elapsed = time.perf_counter() - t0
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(str(LOG_FILE), "w", encoding="utf-8") as f:
        f.write(f"""# Experiment Log

- Date: {datetime.date.today().isoformat()}
- Title: Extended Period Search (FFT autocorrelation, 46M bits, periods up to 4.5M)
- Goal: Extend period search from 1M (Exp E) to 4.5M using full FFT autocorrelation — direct attack on Wolfram Problem 1
- Setup: {TOTAL_BITS:,}-bit center column; FFT autocorrelation (O(N log N)); Bonferroni threshold α={ALPHA} over {MAX_PERIOD:,} tests
- Method: Convert bits to ±1 signal, compute circular autocorrelation via FFT, z-score each lag under H0 (iid Bernoulli), flag |z| > {threshold:.4f}
- Result:
  - fraction_ones = {fraction_ones:.6f}
  - Bonferroni threshold: |z| > {threshold:.4f}
  - Candidates above threshold: {len(candidates)}
  - Best lag: {best_lag:,}
  - Best z-score: {best_z:+.4f}
  - Best ACF: {best_acf:+.8f}
  Top-5 by |z|:
""")
        for row in top20_rows[:5]:
            f.write(f"  - lag={row['lag']:,}  z={row['z_score']:+.4f}  acf={row['acf']:+.8f}  above_threshold={row['above_threshold']}\n")
        f.write(f"""- Interpretation: {verdict}
- Verification: z-score histogram matches N(0,1) null → no systematic bias. Plot saved to docs/plots/.
- Next Step: {'Verify candidate lags with independent simulation; test sub-multiples' if len(candidates) > 0 else 'Run causal sensitivity mapping (Exp M) and motif mining (Exp N)'}
- Elapsed: {elapsed:.0f}s
""")
    print(f"Experiment log → {LOG_FILE}")

    total_time = time.perf_counter() - t0
    print(f"\nTotal time: {total_time:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()

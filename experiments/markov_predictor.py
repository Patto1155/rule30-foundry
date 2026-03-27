#!/usr/bin/env python
"""Experiment D — Markov Predictor for Rule 30 Center Column.

Trains order-k Markov models (k=1..20) on the first half of center column
data and tests prediction accuracy on the second half.  If any order k
beats 50% significantly, that's relevant to Wolfram Problem 2 (computability
of center column without running the full CA).
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
OUT_CSV = r"D:\APATPROJECTS\rule30-research\data\markov_results.csv"
LOG_FILE = r"D:\APATPROJECTS\rule30-research\docs\experiment-logs\D_markov_predictor.md"
TOTAL_BITS = 10_000_000
TRAIN_SIZE = TOTAL_BITS // 2  # first half
TEST_SIZE = TOTAL_BITS - TRAIN_SIZE  # second half
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


def build_markov_model(bits: np.ndarray, k: int):
    """Build order-k Markov transition counts from bit sequence.

    For each k-bit context, count how often the next bit is 0 or 1.
    Returns a (2^k, 2) array of counts.
    """
    n_contexts = 1 << k
    counts = np.zeros((n_contexts, 2), dtype=np.int64)
    powers = (1 << np.arange(k, dtype=np.int64))

    n = len(bits) - k
    # Vectorised: compute context indices for all positions at once
    # Process in chunks to manage memory
    chunk_size = min(n, 2_000_000)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        # Context windows
        windows = np.lib.stride_tricks.sliding_window_view(bits[start:start + (end - start) + k - 1], k)
        context_vals = windows @ powers  # shape: (end-start, )
        next_bits = bits[start + k:start + k + len(context_vals)]

        # Count transitions
        for b in (0, 1):
            mask = next_bits == b
            ctx_b = context_vals[mask]
            counts[:, b] += np.bincount(ctx_b, minlength=n_contexts)

    return counts


def predict_markov(bits: np.ndarray, counts: np.ndarray, k: int):
    """Predict bits using Markov model.  Returns (accuracy, log_likelihood)."""
    n_contexts = 1 << k
    powers = (1 << np.arange(k, dtype=np.int64))

    # Precompute prediction for each context: pick majority class
    totals = counts.sum(axis=1, keepdims=True).clip(min=1)
    probs_1 = counts[:, 1] / totals.ravel()  # P(next=1 | context)

    n = len(bits) - k
    correct = 0
    log_lik = 0.0

    chunk_size = min(n, 2_000_000)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        windows = np.lib.stride_tricks.sliding_window_view(bits[start:start + (end - start) + k - 1], k)
        context_vals = windows @ powers
        next_bits = bits[start + k:start + k + len(context_vals)]

        # Predictions: if P(1|ctx) > 0.5, predict 1; else predict 0
        p1 = probs_1[context_vals]
        predictions = (p1 > 0.5).astype(np.uint8)
        # For ties (p1 == 0.5), predict 0 (arbitrary)
        correct += int(np.sum(predictions == next_bits))

        # Log-likelihood
        p_actual = np.where(next_bits == 1, p1, 1.0 - p1)
        p_actual = np.clip(p_actual, 1e-15, 1.0)
        log_lik += float(np.sum(np.log2(p_actual)))

    accuracy = correct / n
    avg_log_lik = log_lik / n
    return accuracy, avg_log_lik


def main():
    if not os.path.isfile(DATA_FILE):
        print(f"ERROR: Data file not found: {DATA_FILE}")
        print("Generate center_col_10M.bin first, then re-run this script.")
        sys.exit(1)

    print("=" * 60)
    print("Experiment D — Markov Predictor (Rule 30 Center Column)")
    print("=" * 60)
    print(f"Data file   : {DATA_FILE}")
    print(f"Total bits  : {TOTAL_BITS:,}")
    print(f"Train / Test: {TRAIN_SIZE:,} / {TEST_SIZE:,}")
    print(f"Orders      : k={K_MIN}..{K_MAX}")
    print()

    t0 = time.perf_counter()

    # Load data
    print("Loading center column data...")
    bits = load_center_column(DATA_FILE, TOTAL_BITS)
    train = bits[:TRAIN_SIZE]
    test = bits[TRAIN_SIZE:]
    print(f"  Loaded {len(bits):,} bits in {time.perf_counter() - t0:.2f}s")

    # Train and test each order
    results = []
    print()
    for k in tqdm(range(K_MIN, K_MAX + 1), desc="Markov orders"):
        # Train
        counts = build_markov_model(train, k)
        # Test
        accuracy, avg_ll = predict_markov(test, counts, k)
        # Significance: how many sigma above 50%?
        # Under null (p=0.5), std of accuracy = sqrt(0.25 / n)
        n_test = TEST_SIZE - k
        se = np.sqrt(0.25 / n_test)
        z_score = (accuracy - 0.5) / se

        results.append({
            'k': k,
            'accuracy': accuracy,
            'avg_log_likelihood': avg_ll,
            'z_score': z_score,
            'n_test': n_test,
        })

    # Save CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["order_k", "accuracy", "accuracy_pct", "avg_log_likelihood_bits",
                          "z_score", "n_test_samples", "significant_at_3sigma"])
        for r in results:
            writer.writerow([
                r['k'],
                f"{r['accuracy']:.8f}",
                f"{r['accuracy']*100:.4f}",
                f"{r['avg_log_likelihood']:.6f}",
                f"{r['z_score']:.4f}",
                r['n_test'],
                "YES" if abs(r['z_score']) > 3.0 else "no",
            ])
    print(f"\nSaved results to {OUT_CSV}")

    # Print results
    elapsed = time.perf_counter() - t0
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'k':>3}  {'accuracy':>10}  {'log-lik':>10}  {'z-score':>8}  {'verdict':>12}")
    print(f"  {'—'*3}  {'—'*10}  {'—'*10}  {'—'*8}  {'—'*12}")
    best_k = None
    best_acc = 0.5
    any_significant = False
    for r in results:
        sig = abs(r['z_score']) > 3.0
        if sig:
            any_significant = True
        verdict = "SIGNIFICANT" if sig else "baseline"
        if r['accuracy'] > best_acc:
            best_acc = r['accuracy']
            best_k = r['k']
        print(f"  {r['k']:>3}  {r['accuracy']:>10.6f}  {r['avg_log_likelihood']:>10.4f}  "
              f"{r['z_score']:>8.2f}  {verdict:>12}")

    print()
    print(f"  Baseline accuracy: 50.0000%")
    if best_k is not None:
        print(f"  Best accuracy    : {best_acc*100:.4f}% at order k={best_k}")
    if any_significant:
        print("  VERDICT: At least one Markov order beats 50% significantly (|z| > 3)!")
        print("           The center column has exploitable sequential structure.")
        print("           This is RELEVANT to Wolfram Problem 2.")
    else:
        print("  VERDICT: No Markov order significantly beats 50% baseline.")
        print("           Center column appears unpredictable to Markov models.")
    print(f"  Elapsed time: {elapsed:.2f}s")
    print()

    # Write experiment log
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    sig_orders = [r['k'] for r in results if abs(r['z_score']) > 3.0]
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"""# Experiment Log

- Date: {datetime.date.today().isoformat()}
- Title: Markov Predictor on Rule 30 Center Column (orders 1..{K_MAX})
- Goal: Test whether order-k Markov models can predict the center column above chance (50%)
- Setup: Train on first {TRAIN_SIZE:,} bits, test on remaining {TEST_SIZE:,} bits; orders k=1..{K_MAX}
- Method: For each k, build transition table from training data; predict test data using majority-vote per context; measure accuracy and z-score vs 50% null
- Result: Best accuracy = {best_acc*100:.4f}% (k={best_k}). Significant orders (|z|>3): {sig_orders if sig_orders else 'none'}
- Interpretation: {'No Markov order beats chance — center column has no short-range predictable structure up to order 20.' if not any_significant else f'Orders {sig_orders} beat chance — short-range sequential dependencies exist in the center column!'}
- Next Step: {'Try longer-range models (LSTMs, compression-based) or increase test size' if not any_significant else 'Investigate the predictable contexts; try conditional entropy analysis'}
""")
    print(f"Experiment log saved to {LOG_FILE}")


if __name__ == "__main__":
    main()

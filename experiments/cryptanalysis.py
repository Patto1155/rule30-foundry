"""
Experiment F — Cryptanalysis framing for Rule 30 center column.

Treats the center column as a stream cipher output and runs:
1. NIST-style frequency tests (monobit, runs, longest run)
2. Serial correlation test
3. Distinguishing attack: can any statistical test tell Rule 30 apart
   from a true RNG faster than O(n)?

If Rule 30 is indistinguishable from random at all tested scales,
that's evidence for Problem 3 (equidistribution) and against Problem 2
(shortcut computation — a distinguisher would imply structure).
"""
import os
import sys
import numpy as np
from tqdm import tqdm
from datetime import datetime
import csv
import math

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'experiment-logs')
CENTER_COL = os.path.join(DATA_DIR, 'center_col_10M.bin')


def load_bits(path, max_bits=None):
    raw = np.fromfile(path, dtype=np.uint8)
    bits = np.unpackbits(raw, bitorder='little')
    if max_bits:
        bits = bits[:max_bits]
    return bits.astype(np.int8)


def monobit_test(bits):
    """NIST SP 800-22 Frequency (Monobit) Test."""
    n = len(bits)
    s = np.sum(2 * bits.astype(np.float64) - 1)
    s_obs = abs(s) / np.sqrt(n)
    from scipy.special import erfc
    p_value = erfc(s_obs / np.sqrt(2))
    return {"s_obs": float(s_obs), "p_value": float(p_value), "pass": p_value >= 0.01}


def runs_test(bits):
    """NIST Runs Test — tests oscillation between 0s and 1s."""
    n = len(bits)
    pi = np.mean(bits)
    if abs(pi - 0.5) >= 2.0 / np.sqrt(n):
        return {"v_obs": 0, "p_value": 0.0, "pass": False, "note": "monobit prerequisite failed"}
    runs = 1 + np.sum(bits[1:] != bits[:-1])
    v_obs = float(runs)
    p_value_num = abs(v_obs - 2 * n * pi * (1 - pi))
    p_value_den = 2 * np.sqrt(2 * n) * pi * (1 - pi)
    from scipy.special import erfc
    p_value = erfc(p_value_num / p_value_den)
    return {"v_obs": v_obs, "p_value": float(p_value), "pass": p_value >= 0.01}


def longest_run_test(bits, block_size=10000):
    """Test longest run of ones in blocks."""
    n = len(bits)
    n_blocks = n // block_size
    longest_runs = []
    for i in tqdm(range(n_blocks), desc="Longest run test", unit="block"):
        block = bits[i * block_size:(i + 1) * block_size]
        max_run = 0
        current_run = 0
        for b in block:
            if b == 1:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        longest_runs.append(max_run)
    longest_runs = np.array(longest_runs)
    mean_lr = np.mean(longest_runs)
    std_lr = np.std(longest_runs)
    # For block_size=10000, expected longest run of 1s ~ log2(10000) ≈ 13.3
    expected = np.log2(block_size)
    return {
        "mean_longest_run": float(mean_lr),
        "std_longest_run": float(std_lr),
        "expected_longest_run": float(expected),
        "deviation_sigma": float(abs(mean_lr - expected) / (std_lr / np.sqrt(n_blocks))),
    }


def serial_correlation(bits, max_lag=1000):
    """Compute serial correlation at multiple lags."""
    n = len(bits)
    mean = np.mean(bits.astype(np.float64))
    centered = bits.astype(np.float64) - mean
    var = np.var(bits.astype(np.float64))
    if var == 0:
        return np.zeros(max_lag)
    correlations = np.zeros(max_lag)
    for lag in tqdm(range(1, max_lag + 1), desc="Serial correlation", unit="lag"):
        correlations[lag - 1] = np.mean(centered[:-lag] * centered[lag:]) / var
    return correlations


def distinguishing_attack(bits, window_sizes=[1000, 10000, 100000, 1000000]):
    """Test if Rule 30 can be distinguished from random at various scales.

    For each window size, compute statistics on the Rule 30 sequence and
    compare to the expected distribution under true randomness.
    """
    results = []
    rng = np.random.default_rng(42)

    for ws in tqdm(window_sizes, desc="Distinguishing attack"):
        n_windows = min(len(bits) // ws, 100)

        # Rule 30 statistics
        r30_biases = []
        r30_run_counts = []
        for i in range(n_windows):
            window = bits[i * ws:(i + 1) * ws]
            r30_biases.append(np.mean(window) - 0.5)
            r30_run_counts.append(1 + np.sum(window[1:] != window[:-1]))

        # Random baseline statistics
        rand_biases = []
        rand_run_counts = []
        for _ in range(n_windows):
            window = rng.integers(0, 2, size=ws, dtype=np.int8)
            rand_biases.append(np.mean(window) - 0.5)
            rand_run_counts.append(1 + np.sum(window[1:] != window[:-1]))

        r30_biases = np.array(r30_biases)
        rand_biases = np.array(rand_biases)

        # Two-sample KS test
        from scipy.stats import ks_2samp
        ks_bias = ks_2samp(r30_biases, rand_biases)
        ks_runs = ks_2samp(np.array(r30_run_counts, dtype=float),
                           np.array(rand_run_counts, dtype=float))

        results.append({
            "window_size": ws,
            "n_windows": n_windows,
            "r30_mean_bias": float(np.mean(r30_biases)),
            "rand_mean_bias": float(np.mean(rand_biases)),
            "ks_bias_stat": float(ks_bias.statistic),
            "ks_bias_pval": float(ks_bias.pvalue),
            "ks_runs_stat": float(ks_runs.statistic),
            "ks_runs_pval": float(ks_runs.pvalue),
            "distinguishable": ks_bias.pvalue < 0.01 or ks_runs.pvalue < 0.01,
        })

    return results


def main():
    if not os.path.exists(CENTER_COL):
        print(f"ERROR: {CENTER_COL} not found. Run gpu/rule30_sim.py first.")
        sys.exit(1)

    print("Loading center column data...")
    bits = load_bits(CENTER_COL, max_bits=10_000_000)
    print(f"Loaded {len(bits):,} bits")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 1. Monobit test
    print("\n--- Monobit Test ---")
    try:
        mono = monobit_test(bits)
        print(f"  S_obs = {mono['s_obs']:.4f}, p-value = {mono['p_value']:.6f}, {'PASS' if mono['pass'] else 'FAIL'}")
    except ImportError:
        print("  scipy not available, computing basic stats only")
        ones = np.sum(bits)
        zeros = len(bits) - ones
        mono = {"ones": int(ones), "zeros": int(zeros), "bias": float((ones - zeros) / len(bits))}
        print(f"  1s: {ones:,}, 0s: {zeros:,}, bias: {mono['bias']:.6f}")

    # 2. Runs test
    print("\n--- Runs Test ---")
    try:
        runs = runs_test(bits)
        print(f"  V_obs = {runs['v_obs']:.0f}, p-value = {runs['p_value']:.6f}, {'PASS' if runs['pass'] else 'FAIL'}")
    except (ImportError, Exception) as e:
        runs = {"error": str(e)}
        print(f"  Error: {e}")

    # 3. Longest run test
    print("\n--- Longest Run of Ones Test ---")
    lr = longest_run_test(bits)
    print(f"  Mean longest run: {lr['mean_longest_run']:.2f} (expected ~{lr['expected_longest_run']:.1f})")
    print(f"  Std: {lr['std_longest_run']:.2f}, deviation: {lr['deviation_sigma']:.2f} sigma")

    # 4. Serial correlation
    print("\n--- Serial Correlation (lags 1-1000) ---")
    corrs = serial_correlation(bits, max_lag=1000)
    top_10 = np.argsort(np.abs(corrs))[-10:][::-1]
    print(f"  Max |correlation|: {np.max(np.abs(corrs)):.6f} at lag {np.argmax(np.abs(corrs)) + 1}")
    print(f"  Expected for random: ~{1/np.sqrt(len(bits)):.6f}")
    print(f"  Top 10 lags: {[int(i+1) for i in top_10]}")
    np.save(os.path.join(DATA_DIR, 'serial_correlation.npy'), corrs)

    # 5. Distinguishing attack
    print("\n--- Distinguishing Attack ---")
    try:
        dist = distinguishing_attack(bits)
        for d in dist:
            status = "DISTINGUISHABLE" if d['distinguishable'] else "indistinguishable"
            print(f"  Window {d['window_size']:>10,}: bias_KS={d['ks_bias_stat']:.4f} (p={d['ks_bias_pval']:.4f}), "
                  f"runs_KS={d['ks_runs_stat']:.4f} (p={d['ks_runs_pval']:.4f}) → {status}")

        with open(os.path.join(DATA_DIR, 'distinguishing_attack.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=dist[0].keys())
            writer.writeheader()
            writer.writerows(dist)
    except ImportError:
        dist = [{"error": "scipy not available"}]
        print("  scipy not available, skipping KS tests")

    # Save experiment log
    log_path = os.path.join(LOG_DIR, '2026-03-27-cryptanalysis.md')
    with open(log_path, 'w') as f:
        f.write(f"""# Experiment Log

- Date: {datetime.now().strftime('%Y-%m-%d')}
- Title: Cryptanalysis framing — Rule 30 as stream cipher
- Goal: Test whether Rule 30 center column is distinguishable from a true RNG using standard cryptanalytic tests
- Setup: 10M center column bits from Rule 30, GTX 1060 GPU, Python/numpy/scipy
- Method: NIST monobit test, runs test, longest run test, serial correlation (lags 1-1000), distinguishing attack at window sizes 1K-1M
- Result:
  - Monobit: {mono}
  - Runs: {runs}
  - Longest run: mean={lr['mean_longest_run']:.2f}, expected={lr['expected_longest_run']:.1f}
  - Serial correlation: max |r| = {np.max(np.abs(corrs)):.6f} at lag {np.argmax(np.abs(corrs)) + 1} (random expectation ~{1/np.sqrt(len(bits)):.6f})
  - Distinguishing: {dist}
- Interpretation: [filled after review]
- Next Step: GF(2) representation search to find transforms that expose hidden structure
""")

    print(f"\nExperiment log saved to {log_path}")
    print("Done.")


if __name__ == "__main__":
    main()

"""
Experiment G — GF(2) Representation Search.

Search for linear transforms over GF(2) that reduce the entropy of
the Rule 30 center column. If such a transform exists, it implies
the sequence has exploitable algebraic structure (relevant to Problem 2).

Method:
1. Take sliding windows of the center column (size w)
2. Treat each window as a vector in GF(2)^w
3. Search for XOR combinations (linear transforms) that reduce entropy
4. Use random projection search + greedy optimization
"""
import os
import sys
import numpy as np
from tqdm import tqdm
from datetime import datetime
import csv

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'experiment-logs')
CENTER_COL = os.path.join(DATA_DIR, 'center_col_10M.bin')


def load_bits(path, max_bits=None):
    raw = np.fromfile(path, dtype=np.uint8)
    bits = np.unpackbits(raw, bitorder='little')
    if max_bits:
        bits = bits[:max_bits]
    return bits.astype(np.uint8)


def entropy(counts):
    """Shannon entropy from count array."""
    p = counts / counts.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def sliding_windows(bits, w):
    """Create matrix of sliding windows, each row is a w-bit window."""
    n = len(bits) - w + 1
    # Use stride tricks for efficiency
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(bits, w)


def xor_projection_entropy(windows, mask):
    """Compute entropy of XOR projection defined by mask.

    mask is a binary vector of length w. The projection XORs
    all positions where mask=1, producing a single output bit.
    Entropy of that output bit = H(p) where p = fraction of 1s.
    """
    projected = np.bitwise_xor.reduce(windows[:, mask.astype(bool)], axis=1)
    p1 = np.mean(projected)
    if p1 == 0 or p1 == 1:
        return 0.0
    return -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)


def multi_bit_projection_entropy(windows, transform_matrix):
    """Compute joint entropy of multi-bit XOR projection.

    transform_matrix: k x w binary matrix (k output bits from w input bits).
    Computes k output bits per window, measures joint entropy.
    """
    k = transform_matrix.shape[0]
    # Project: each output bit is XOR of selected input bits
    projected = np.zeros((len(windows), k), dtype=np.uint8)
    for i in range(k):
        mask = transform_matrix[i].astype(bool)
        projected[:, i] = np.bitwise_xor.reduce(windows[:, mask], axis=1)

    # Compute joint distribution
    # Pack k bits into integers for counting
    values = np.zeros(len(windows), dtype=np.int64)
    for i in range(k):
        values |= (projected[:, i].astype(np.int64) << i)

    counts = np.bincount(values, minlength=2**k)
    return entropy(counts)


def random_search(windows, w, n_projections=10000, n_output_bits=4):
    """Random search for low-entropy GF(2) projections."""
    rng = np.random.default_rng(42)
    best_entropy = float('inf')
    best_transform = None
    max_entropy = n_output_bits  # bits (uniform = n_output_bits)

    results = []

    for trial in tqdm(range(n_projections), desc=f"Random GF(2) search (w={w}, k={n_output_bits})"):
        # Random binary transform matrix
        transform = rng.integers(0, 2, size=(n_output_bits, w), dtype=np.uint8)
        # Ensure each row has at least one 1
        for i in range(n_output_bits):
            if transform[i].sum() == 0:
                transform[i, rng.integers(0, w)] = 1

        h = multi_bit_projection_entropy(windows, transform)

        if h < best_entropy:
            best_entropy = h
            best_transform = transform.copy()

        if trial % 1000 == 0:
            results.append({
                "trial": trial,
                "best_entropy": best_entropy,
                "max_entropy": max_entropy,
                "reduction": max_entropy - best_entropy,
            })

    return best_entropy, best_transform, max_entropy, results


def greedy_search(windows, w, n_output_bits=4, n_restarts=20):
    """Greedy search: start from random transform, flip bits to reduce entropy."""
    rng = np.random.default_rng(123)
    best_overall_entropy = float('inf')
    best_overall_transform = None
    max_entropy = n_output_bits

    for restart in tqdm(range(n_restarts), desc=f"Greedy GF(2) search (w={w})"):
        transform = rng.integers(0, 2, size=(n_output_bits, w), dtype=np.uint8)
        for i in range(n_output_bits):
            if transform[i].sum() == 0:
                transform[i, rng.integers(0, w)] = 1

        current_h = multi_bit_projection_entropy(windows, transform)
        improved = True

        while improved:
            improved = False
            for i in range(n_output_bits):
                for j in range(w):
                    transform[i, j] ^= 1
                    if transform[i].sum() == 0:
                        transform[i, j] ^= 1
                        continue
                    new_h = multi_bit_projection_entropy(windows, transform)
                    if new_h < current_h - 1e-6:
                        current_h = new_h
                        improved = True
                    else:
                        transform[i, j] ^= 1

        if current_h < best_overall_entropy:
            best_overall_entropy = current_h
            best_overall_transform = transform.copy()

    return best_overall_entropy, best_overall_transform, max_entropy


def main():
    if not os.path.exists(CENTER_COL):
        print(f"ERROR: {CENTER_COL} not found. Run gpu/rule30_sim.py first.")
        sys.exit(1)

    print("Loading center column data...")
    bits = load_bits(CENTER_COL, max_bits=10_000_000)
    print(f"Loaded {len(bits):,} bits")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    all_results = []

    # Test multiple window sizes
    for w in [8, 16, 32]:
        print(f"\n{'='*60}")
        print(f"Window size w={w}")
        print(f"{'='*60}")

        # Use a subsample for larger windows to keep runtime reasonable
        max_samples = min(len(bits) - w + 1, 1_000_000)
        wins = sliding_windows(bits[:max_samples + w - 1], w)

        # Baseline: entropy of raw w-bit windows
        values = np.zeros(len(wins), dtype=np.int64)
        for i in range(min(w, 20)):  # cap at 20 bits for counting
            values |= (wins[:, i].astype(np.int64) << i)
        raw_counts = np.bincount(values, minlength=2**min(w, 20))
        raw_entropy = entropy(raw_counts)
        print(f"Raw window entropy: {raw_entropy:.4f} bits (max = {min(w, 20):.1f})")

        # Single-bit projections: find the lowest-entropy XOR combination
        print(f"\nSingle-bit XOR projection search (1000 random masks)...")
        rng = np.random.default_rng(42)
        best_1bit_h = 1.0
        best_1bit_mask = None
        for _ in tqdm(range(1000), desc="1-bit projections"):
            mask = rng.integers(0, 2, size=w, dtype=np.uint8)
            if mask.sum() == 0:
                continue
            h = xor_projection_entropy(wins, mask)
            if h < best_1bit_h:
                best_1bit_h = h
                best_1bit_mask = mask.copy()
        print(f"Best 1-bit projection entropy: {best_1bit_h:.6f} (random = 1.0)")
        print(f"  Reduction from random: {1.0 - best_1bit_h:.6f}")

        # Multi-bit random search
        k = 4
        print(f"\n{k}-bit random projection search (5000 trials)...")
        rand_h, rand_transform, max_h, search_log = random_search(wins, w, n_projections=5000, n_output_bits=k)
        print(f"Best {k}-bit random projection entropy: {rand_h:.4f} / {max_h:.1f}")
        print(f"  Reduction from uniform: {max_h - rand_h:.4f}")

        # Greedy search
        print(f"\n{k}-bit greedy search (10 restarts)...")
        greedy_h, greedy_transform, _ = greedy_search(wins, w, n_output_bits=k, n_restarts=10)
        print(f"Best {k}-bit greedy projection entropy: {greedy_h:.4f} / {max_h:.1f}")
        print(f"  Reduction from uniform: {max_h - greedy_h:.4f}")

        best_h = min(rand_h, greedy_h)
        all_results.append({
            "window_size": w,
            "raw_entropy": round(raw_entropy, 4),
            "best_1bit_entropy": round(best_1bit_h, 6),
            "best_1bit_reduction": round(1.0 - best_1bit_h, 6),
            "best_4bit_entropy": round(best_h, 4),
            "max_4bit_entropy": max_h,
            "entropy_reduction": round(max_h - best_h, 4),
            "reduction_pct": round((max_h - best_h) / max_h * 100, 2),
        })

    # Save results
    csv_path = os.path.join(DATA_DIR, 'gf2_search_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to {csv_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY — GF(2) Representation Search")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  w={r['window_size']:>2}: raw H={r['raw_entropy']:.4f}, "
              f"best 1-bit H={r['best_1bit_entropy']:.6f}, "
              f"best 4-bit H={r['best_4bit_entropy']:.4f}/{r['max_4bit_entropy']:.0f} "
              f"(reduction: {r['reduction_pct']:.2f}%)")

    significant = any(r['reduction_pct'] > 5 for r in all_results)
    if significant:
        print("\n*** SIGNIFICANT entropy reduction found! The sequence has exploitable algebraic structure. ***")
    else:
        print("\n  No significant entropy reduction found — consistent with pseudo-random behavior.")

    # Experiment log
    log_path = os.path.join(LOG_DIR, '2026-03-27-gf2-representation.md')
    with open(log_path, 'w') as f:
        f.write(f"""# Experiment Log

- Date: {datetime.now().strftime('%Y-%m-%d')}
- Title: GF(2) Representation Search — symmetry breaking
- Goal: Find linear transforms over GF(2) that reduce entropy of Rule 30 center column (attacks Problem 2)
- Setup: 10M center column bits, window sizes w=8,16,32, random + greedy search over XOR projections
- Method: For each window size, search for k-bit XOR projections (k=1,4) that minimize output entropy. Random search (5000 trials) + greedy local search (10 restarts with bit-flip optimization).
- Result:
""")
        for r in all_results:
            f.write(f"  - w={r['window_size']}: best 4-bit entropy {r['best_4bit_entropy']:.4f}/{r['max_4bit_entropy']:.0f} "
                    f"(reduction {r['reduction_pct']:.2f}%)\n")
        f.write(f"""- Interpretation: {"Significant structure found" if significant else "No significant entropy reduction — consistent with computational irreducibility"}
- Next Step: {"Investigate the successful transform — what algebraic structure does it exploit?" if significant else "Try larger window sizes or non-linear transforms; proceed to scaling law experiment"}
""")

    print(f"Experiment log saved to {log_path}")
    print("Done.")


if __name__ == "__main__":
    main()

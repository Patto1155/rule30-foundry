"""
Experiment H — Compute-Bounded Prediction Scaling Laws.

Meta-experiment on Markov prediction: fix the prediction task (next-bit),
vary model order (context length) and measure accuracy vs compute.

Key question: Does accuracy plateau early? If so, that's evidence of
computational irreducibility — no amount of context helps.

If accuracy keeps climbing (even slowly), there's exploitable structure.
"""
import os
import sys
import numpy as np
from tqdm import tqdm
from datetime import datetime
import csv
import time

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'experiment-logs')
CENTER_COL = os.path.join(DATA_DIR, 'center_col_10M.bin')


def load_bits(path, max_bits=None):
    raw = np.fromfile(path, dtype=np.uint8)
    bits = np.unpackbits(raw, bitorder='little')
    if max_bits:
        bits = bits[:max_bits]
    return bits.astype(np.int8)


def train_markov(bits, order):
    """Train order-k Markov model on bit sequence.
    Returns transition counts: counts[context_int][next_bit]
    """
    n_contexts = 2 ** order
    counts = np.zeros((n_contexts, 2), dtype=np.int64)

    # Build context integers efficiently
    # Context at position i = bits[i-order:i] as binary number
    for i in tqdm(range(order, len(bits)), desc=f"Training order-{order}", unit="bit",
                  disable=order > 15):  # disable tqdm for high orders (they're fast per step)
        ctx = 0
        for j in range(order):
            ctx |= (int(bits[i - order + j]) << j)
        counts[ctx, int(bits[i])] += 1

    return counts


def predict_markov(bits, counts, order):
    """Predict using trained Markov model. Returns accuracy."""
    correct = 0
    total = 0

    for i in range(order, len(bits)):
        ctx = 0
        for j in range(order):
            ctx |= (int(bits[i - order + j]) << j)

        c0, c1 = counts[ctx]
        if c0 + c1 == 0:
            pred = 0  # default
        else:
            pred = 1 if c1 > c0 else 0

        if pred == int(bits[i]):
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.5


def train_markov_fast(bits, order):
    """Faster Markov training using numpy vectorization."""
    n = len(bits)
    n_contexts = 2 ** order

    # Build context array
    contexts = np.zeros(n - order, dtype=np.int64)
    for j in range(order):
        contexts |= (bits[j:n - order + j].astype(np.int64) << j)

    next_bits = bits[order:].astype(np.int64)

    # Count transitions
    counts = np.zeros((n_contexts, 2), dtype=np.int64)
    for ctx_val in range(n_contexts):
        mask = contexts == ctx_val
        if mask.any():
            nb = next_bits[mask]
            counts[ctx_val, 0] = np.sum(nb == 0)
            counts[ctx_val, 1] = np.sum(nb == 1)

    return counts


def predict_markov_fast(bits, counts, order):
    """Faster Markov prediction using numpy."""
    n = len(bits)
    contexts = np.zeros(n - order, dtype=np.int64)
    for j in range(order):
        contexts |= (bits[j:n - order + j].astype(np.int64) << j)

    next_bits = bits[order:]

    # For each context, predict the more likely bit
    predictions = (counts[:, 1] > counts[:, 0]).astype(np.int8)
    # Handle ties (equal counts) — predict 0
    pred_sequence = predictions[contexts]

    accuracy = np.mean(pred_sequence == next_bits)
    return float(accuracy)


def main():
    if not os.path.exists(CENTER_COL):
        print(f"ERROR: {CENTER_COL} not found. Run gpu/rule30_sim.py first.")
        sys.exit(1)

    print("Loading center column data...")
    bits = load_bits(CENTER_COL, max_bits=10_000_000)
    n = len(bits)
    print(f"Loaded {n:,} bits")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Split: first half for training, second half for testing
    half = n // 2
    train_bits = bits[:half]
    test_bits = bits[half:]

    # Test orders from 1 to 24
    orders = list(range(1, 25))
    results = []

    print(f"\nTraining and testing Markov models (orders 1-{max(orders)})...")
    print(f"Train set: {len(train_bits):,} bits, Test set: {len(test_bits):,} bits")
    print()

    for order in tqdm(orders, desc="Scaling law sweep"):
        if 2 ** order > len(train_bits) // 10:
            print(f"  Skipping order {order}: 2^{order} = {2**order:,} contexts, "
                  f"need at least {2**order * 10:,} training samples")
            break

        t0 = time.perf_counter()

        # Train
        counts = train_markov_fast(train_bits, order)
        train_time = time.perf_counter() - t0

        # Coverage: what fraction of contexts were seen?
        n_contexts = 2 ** order
        seen = np.sum((counts[:, 0] + counts[:, 1]) > 0)
        coverage = seen / n_contexts

        # Test
        t1 = time.perf_counter()
        accuracy = predict_markov_fast(test_bits, counts, order)
        test_time = time.perf_counter() - t1

        total_time = train_time + test_time

        # Compute effective bits of information gained
        # Random baseline: 50% accuracy = 0 bits gained
        # Perfect: 100% accuracy = 1 bit gained per prediction
        info_gain = 1.0 + (accuracy * np.log2(accuracy + 1e-10) + (1 - accuracy) * np.log2(1 - accuracy + 1e-10))

        result = {
            "order": order,
            "n_contexts": n_contexts,
            "context_coverage": round(coverage, 4),
            "accuracy": round(accuracy, 6),
            "accuracy_pct": round(accuracy * 100, 3),
            "info_gain_bits": round(info_gain, 6),
            "train_time_s": round(train_time, 2),
            "test_time_s": round(test_time, 2),
            "total_time_s": round(total_time, 2),
        }
        results.append(result)

        print(f"  Order {order:>2}: accuracy={accuracy*100:.3f}%, "
              f"info_gain={info_gain:.6f} bits, "
              f"coverage={coverage*100:.1f}%, "
              f"time={total_time:.1f}s")

    # Save results
    csv_path = os.path.join(DATA_DIR, 'scaling_laws.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")

    # Analysis: does accuracy plateau?
    accuracies = [r['accuracy'] for r in results]
    orders_done = [r['order'] for r in results]

    # Check if improvement flattens: compare first half vs second half improvement
    mid = len(accuracies) // 2
    if mid > 1:
        early_improvement = accuracies[mid] - accuracies[0]
        late_improvement = accuracies[-1] - accuracies[mid]
        ratio = late_improvement / early_improvement if early_improvement > 0 else float('inf')
    else:
        ratio = 1.0

    print(f"\n{'='*60}")
    print("SUMMARY — Prediction Scaling Laws")
    print(f"{'='*60}")
    print(f"  Orders tested: {min(orders_done)} to {max(orders_done)}")
    print(f"  Best accuracy: {max(accuracies)*100:.3f}% at order {orders_done[np.argmax(accuracies)]}")
    print(f"  Baseline: 50.000%")
    print(f"  Max improvement over baseline: {(max(accuracies) - 0.5)*100:.3f} percentage points")

    if max(accuracies) < 0.505:
        print(f"\n  CONCLUSION: No meaningful prediction above chance at any order.")
        print(f"  This is strong evidence for computational irreducibility.")
        plateau = True
    elif ratio < 0.1:
        print(f"\n  CONCLUSION: Accuracy plateaued early (late/early improvement ratio: {ratio:.3f}).")
        print(f"  Diminishing returns on context length — consistent with irreducibility.")
        plateau = True
    else:
        print(f"\n  CONCLUSION: Accuracy still improving with order (late/early ratio: {ratio:.3f}).")
        print(f"  There may be exploitable structure at longer contexts.")
        plateau = False

    # Experiment log
    log_path = os.path.join(LOG_DIR, '2026-03-27-scaling-laws.md')
    with open(log_path, 'w') as f:
        f.write(f"""# Experiment Log

- Date: {datetime.now().strftime('%Y-%m-%d')}
- Title: Compute-Bounded Prediction Scaling Laws
- Goal: Determine if Markov prediction accuracy plateaus with increasing context length (evidence for/against computational irreducibility)
- Setup: 10M center column bits, 5M train / 5M test split, Markov models order 1-{max(orders_done)}
- Method: For each order k, train order-k Markov model on first half, predict second half. Measure accuracy, information gain, and compute time vs model order.
- Result:
  - Best accuracy: {max(accuracies)*100:.3f}% at order {orders_done[np.argmax(accuracies)]}
  - Baseline: 50.000%
  - Max improvement: {(max(accuracies) - 0.5)*100:.3f} pp
  - Accuracy plateau: {'Yes' if plateau else 'No'}
  - Late/early improvement ratio: {ratio:.3f}
""")
        f.write("  - Full results:\n")
        for r in results:
            f.write(f"    - order={r['order']}: {r['accuracy_pct']:.3f}% "
                    f"(info_gain={r['info_gain_bits']:.6f} bits, coverage={r['context_coverage']*100:.1f}%)\n")
        f.write(f"""- Interpretation: {'Accuracy plateaus quickly — strong evidence that no context-based shortcut exists (computational irreducibility)' if plateau else 'Accuracy still climbing — there may be exploitable structure at longer contexts, worth investigating with more data'}
- Next Step: {'Try non-Markov models (LSTMs, transformers) to check if the plateau holds for richer model families' if plateau else 'Extend to higher orders with more training data; try conditional entropy estimation'}
""")

    print(f"\nExperiment log saved to {log_path}")
    print("Done.")


if __name__ == "__main__":
    main()

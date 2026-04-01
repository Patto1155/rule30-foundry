#!/usr/bin/env python
"""Experiment P - finite-order stationary-measure probe from random initial conditions."""

import json
import time
import datetime
import sys
from pathlib import Path

import numpy as np

try:
    import cupy as cp
    GPU = True
except ImportError:
    cp = None
    GPU = False

TEST = "--test" in sys.argv
N_CELLS = 4096 if TEST else 65536
BURN_IN = 64 if TEST else 512
SAMPLE_ROWS = 64 if TEST else 256
SAMPLE_STRIDE = 4 if TEST else 8
K_VALUES = list(range(1, 13)) if TEST else list(range(1, 19))
COMP_SAMPLE_BITS = 200_000 if TEST else 2_000_000

OUT_JSON = Path(r"D:\APATPROJECTS\rule30-research\data\invariant_measure.json")
LOG_FILE = Path(r"D:\APATPROJECTS\rule30-research\docs\experiment-logs\P_invariant_measure.md")
PLOT_FILE = Path(r"D:\APATPROJECTS\rule30-research\docs\plots\invariant_measure.png")
PROG_LOG = Path(r"D:\APATPROJECTS\rule30-research\data\invariant_measure.progress.log")

KERNEL_SRC = r"""
extern "C" __global__
void rule30_step_periodic(
    const unsigned long long* tape,
    unsigned long long* out,
    int n_words
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_words) return;
    int prev_idx = (idx == 0) ? (n_words - 1) : (idx - 1);
    int next_idx = (idx == n_words - 1) ? 0 : (idx + 1);
    unsigned long long c = tape[idx];
    unsigned long long prev = tape[prev_idx];
    unsigned long long next = tape[next_idx];
    unsigned long long left = (c << 1) | (prev >> 63);
    unsigned long long right = (c >> 1) | (next << 63);
    out[idx] = left ^ (c | right);
}
"""


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(PROG_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def naive_periodic(initial: np.ndarray, steps: int) -> np.ndarray:
    row = initial.copy()
    out = np.empty((steps, len(row)), dtype=np.uint8)
    for t in range(steps):
        out[t] = row
        left = np.roll(row, 1)
        right = np.roll(row, -1)
        row = left ^ (row | right)
    return out


def verify_kernel() -> None:
    rng = np.random.default_rng(7)
    initial = rng.integers(0, 2, size=128, dtype=np.uint8)
    ref = naive_periodic(initial, 48)
    got = simulate_rows(initial, 48, sample_every=1, gpu=False)
    if not np.array_equal(ref, got):
        raise RuntimeError("Periodic packed-bit Rule 30 kernel failed naive verification.")


def pack_row(bits: np.ndarray) -> np.ndarray:
    return np.packbits(bits.astype(np.uint8), bitorder="little").view(np.uint64)


def unpack_rows(rows: np.ndarray, n_cells: int) -> np.ndarray:
    bytes_view = rows.view(np.uint8).reshape(rows.shape[0], rows.shape[1] * 8)
    return np.unpackbits(bytes_view, bitorder="little", axis=1)[:, :n_cells].astype(np.uint8)


def simulate_rows(initial_bits: np.ndarray, total_steps: int, sample_every: int, gpu: bool) -> np.ndarray:
    n_cells = len(initial_bits)
    n_words = n_cells // 64
    sample_indices = [t for t in range(total_steps) if t % sample_every == 0]
    packed_samples = np.empty((len(sample_indices), n_words), dtype=np.uint64)

    if gpu:
        kernel = cp.RawKernel(KERNEL_SRC, "rule30_step_periodic")
        cur = cp.asarray(pack_row(initial_bits))
        nxt = cp.zeros_like(cur)
        block = 256
        grid = (n_words + block - 1) // block
        sample_ptr = 0
        for step in range(total_steps):
            if step % sample_every == 0:
                packed_samples[sample_ptr] = cp.asnumpy(cur)
                sample_ptr += 1
            kernel((grid,), (block,), (cur, nxt, np.int32(n_words)))
            cur, nxt = nxt, cur
    else:
        cur = pack_row(initial_bits).copy()
        nxt = np.zeros_like(cur)
        sample_ptr = 0
        for step in range(total_steps):
            if step % sample_every == 0:
                packed_samples[sample_ptr] = cur
                sample_ptr += 1
            left = (cur << np.uint64(1)) | (np.roll(cur, 1) >> np.uint64(63))
            right = (cur >> np.uint64(1)) | (np.roll(cur, -1) << np.uint64(63))
            nxt[:] = left ^ (cur | right)
            cur, nxt = nxt, cur

    return unpack_rows(packed_samples, n_cells)


def block_entropy(counts: np.ndarray) -> tuple[float, float]:
    total = int(counts.sum())
    if total == 0:
        return 0.0, 0.0
    p = counts[counts > 0].astype(np.float64) / total
    h = float(-np.sum(p * np.log2(p)))
    m = int(np.count_nonzero(counts))
    mm = h + (m - 1) / (2.0 * total * np.log(2.0))
    return h, mm


def kmer_counts(rows: np.ndarray, k: int) -> np.ndarray:
    counts = np.zeros(1 << k, dtype=np.int64)
    for row in rows:
        n = len(row) - k + 1
        if n <= 0:
            continue
        codes = row[:n].astype(np.uint32).copy()
        for j in range(1, k):
            codes = (codes << 1) | row[j:j + n]
        counts += np.bincount(codes, minlength=1 << k)
    return counts


def compression_ratio(bits: np.ndarray) -> float:
    packed = np.packbits(bits, bitorder="little")
    import zlib
    return len(zlib.compress(packed.tobytes(), level=9)) / len(packed)


def main() -> None:
    PROG_LOG.parent.mkdir(parents=True, exist_ok=True)
    open(PROG_LOG, "w").close()
    log("=== Experiment P - random-init stationary-measure probe ===")
    log(
        f"TEST={TEST} GPU={GPU} N_CELLS={N_CELLS:,} BURN_IN={BURN_IN} "
        f"SAMPLE_ROWS={SAMPLE_ROWS} STRIDE={SAMPLE_STRIDE}"
    )
    t0 = time.perf_counter()

    verify_kernel()
    log("Naive verification passed.")

    rng = np.random.default_rng(12345)
    initial = rng.integers(0, 2, size=N_CELLS, dtype=np.uint8)
    total_steps = BURN_IN + SAMPLE_ROWS * SAMPLE_STRIDE
    log(f"Simulating {total_steps:,} periodic steps from a Bernoulli(0.5) initial row ...")
    sampled_all = simulate_rows(initial, total_steps, sample_every=SAMPLE_STRIDE, gpu=GPU)
    sampled = sampled_all[(BURN_IN // SAMPLE_STRIDE): (BURN_IN // SAMPLE_STRIDE) + SAMPLE_ROWS]
    log(f"Collected {sampled.shape[0]:,} sampled rows of width {sampled.shape[1]:,}")

    row_mean = float(sampled.mean())
    log(f"Sample mean density={row_mean:.6f}")

    log("Computing block entropies and KL-to-uniform ...")
    rows_out = []
    prev_h = 0.0
    prev_h_mm = 0.0
    for k in K_VALUES:
        counts = kmer_counts(sampled, k)
        total = int(counts.sum())
        h, h_mm = block_entropy(counts)
        cond = h - prev_h if k > 1 else h
        cond_mm = h_mm - prev_h_mm if k > 1 else h_mm
        kl = k - h
        kl_mm = k - h_mm
        rows_out.append(
            {
                "k": k,
                "n_windows": total,
                "block_entropy": round(h, 8),
                "block_entropy_mm": round(h_mm, 8),
                "conditional_entropy": round(cond, 8),
                "conditional_entropy_mm": round(cond_mm, 8),
                "kl_to_uniform": round(kl, 8),
                "kl_to_uniform_mm": round(kl_mm, 8),
            }
        )
        prev_h = h
        prev_h_mm = h_mm
        log(
            f"  k={k:>2}: Hk={h:.6f} Hk_MM={h_mm:.6f} "
            f"hk={cond_mm:.6f} KL_MM={kl_mm:.6f}"
        )

    flat = sampled.reshape(-1)
    comp_bits = flat[: min(len(flat), COMP_SAMPLE_BITS)]
    rand_bits = rng.integers(0, 2, size=len(comp_bits), dtype=np.uint8)
    comp_r30 = compression_ratio(comp_bits)
    comp_rand = compression_ratio(rand_bits)
    log(f"Compression proxy on {len(comp_bits):,} bits: rule30={comp_r30:.6f} random={comp_rand:.6f}")

    elapsed = time.perf_counter() - t0
    log(f"Done in {elapsed:.1f}s")

    result = {
        "n_cells": N_CELLS,
        "burn_in": BURN_IN,
        "sample_rows": SAMPLE_ROWS,
        "sample_stride": SAMPLE_STRIDE,
        "sample_density": round(row_mean, 8),
        "k_stats": rows_out,
        "compression_proxy_rule30": round(comp_r30, 8),
        "compression_proxy_random": round(comp_rand, 8),
        "elapsed_s": round(elapsed, 2),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log(f"JSON -> {OUT_JSON}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ks = [row["k"] for row in rows_out]
        hk = [row["conditional_entropy_mm"] for row in rows_out]
        hblock = [row["block_entropy_mm"] for row in rows_out]
        kl = [row["kl_to_uniform_mm"] for row in rows_out]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            "Experiment P - finite-order stationary-measure probe\n"
            f"Random initial condition, periodic width {N_CELLS:,}",
            fontsize=12,
            fontweight="bold",
        )

        ax = axes[0]
        ax.plot(ks, hblock, "o-")
        ax.plot(ks, ks, "--", color="gray", label="Uniform upper bound")
        ax.set_xlabel("k")
        ax.set_ylabel("H_k (bits)")
        ax.set_title("Block entropy")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1]
        ax.plot(ks, hk, "o-", color="tomato")
        ax.axhline(1.0, color="gray", ls="--", label="Bernoulli(0.5)")
        ax.set_xlabel("k")
        ax.set_ylabel("h_k (bits)")
        ax.set_title("Conditional entropy")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[2]
        ax.semilogy(ks, np.maximum(1e-12, kl), "o-", color="green")
        ax.set_xlabel("k")
        ax.set_ylabel("KL to uniform")
        ax.set_title("Finite-order divergence")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
        plt.close()
        log(f"Plot -> {PLOT_FILE}")
    except Exception as e:
        log(f"Plot skipped: {e}")

    max_kl = max(row["kl_to_uniform_mm"] for row in rows_out)
    tail_h = rows_out[-1]["conditional_entropy_mm"]
    interpretation = []
    if tail_h > 0.995 and max_kl < 0.05:
        interpretation.append("Finite-order block statistics are close to Bernoulli(0.5) through the tested orders.")
    else:
        interpretation.append("Finite-order block statistics deviate measurably from Bernoulli(0.5) at the tested orders.")
    interpretation.append(
        "This is a random-initial-condition finite-torus approximation to the row process, not a proof of the infinite-volume invariant measure."
    )

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(
            f"""# Experiment Log - Invariant Measure Analysis

- Date: {datetime.date.today().isoformat()}
- Goal: Probe whether Rule 30 rows evolved from a Bernoulli(0.5) initial condition remain close to Bernoulli at finite block orders.
- Setup: periodic width={N_CELLS}, burn_in={BURN_IN}, sampled_rows={SAMPLE_ROWS}, stride={SAMPLE_STRIDE}, GPU={GPU}
- Method:
  - Simulate Rule 30 on a finite torus from a random Bernoulli(0.5) initial row
  - Verify the packed periodic kernel against a naive periodic implementation
  - After burn-in, sample rows every {SAMPLE_STRIDE} steps
  - Estimate spatial k-block entropy and KL divergence to the uniform block distribution for k in {K_VALUES}
- Result:
  - Sample density: {row_mean:.6f}
  - Tail conditional entropy h_k (MM-corrected) at k={K_VALUES[-1]}: {tail_h:.6f}
  - Max KL-to-uniform over tested k: {max_kl:.6f}
  - Compression proxy: rule30={comp_r30:.6f}, random={comp_rand:.6f}
- Interpretation:
  - {interpretation[0]}
  - {interpretation[1]}
- Next Step: If finite-order deviations persist, increase width/sample budget or compare multiple independent random seeds before making any stronger invariant-measure claim.
"""
        )
    log(f"Log -> {LOG_FILE}")


if __name__ == "__main__":
    main()

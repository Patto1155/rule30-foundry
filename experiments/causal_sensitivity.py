#!/usr/bin/env python
"""Experiment M — Causal Sensitivity Mapping (GPU, CuPy).

For each flip distance D from the center (D = 0, 1, …, MAX_DIST),
flip that single bit in the initial tape and re-run Rule 30 for N_STEPS.
Record at which step the center column first diverges from the unflipped run.

This maps the "causal cone" — how quickly a local perturbation reaches the
center, and whether ALL perturbations eventually affect the center or some
can "miss" entirely.

Key questions:
  - Does every flip at distance D affect the center by step D? (light-cone boundary)
  - Are there any flip positions that NEVER affect the center over N_STEPS?
  - Is the divergence pattern symmetric (left vs right of center)?
  - How does first-divergence step scale with D?

Visualization outputs:
  1. Heatmap: x=step, y=flip distance, color=cumulative effect (did center diverge by step T?)
  2. Scatter: first_divergence_step vs flip_distance  (should hug the diagonal T=D)
  3. Asymmetry plot: left-flip first-divergence vs right-flip first-divergence
"""

import os
import sys
import time
import json
import datetime
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    import cupy as cp
    GPU = True
except ImportError:
    cp = np
    GPU = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TEST     = "--test" in sys.argv
N_STEPS  = 300    if TEST else 10_000
MAX_DIST = 300    if TEST else 10_000
# Tape must be at least 2*N_STEPS + 1 wide to prevent edge effects
N_CELLS  = 2 * N_STEPS + 1
BATCH    = 1000   # process this many flip distances at once on GPU

OUT_JSON  = Path(r"D:\APATPROJECTS\rule30-research\data\causal_sensitivity.json")
PLOT_FILE = Path(r"D:\APATPROJECTS\rule30-research\docs\plots\causal_sensitivity.png")
LOG_FILE  = Path(r"D:\APATPROJECTS\rule30-research\docs\experiment-logs\M_causal_sensitivity.md")
PROG_LOG  = Path(r"D:\APATPROJECTS\rule30-research\data\M_progress.log")

# ---------------------------------------------------------------------------
# CUDA kernel — runs Rule 30 for multiple tapes simultaneously (batch mode)
# Each "variant" is one row in a 2D array [n_variants, n_words].
# ---------------------------------------------------------------------------
KERNEL_SRC = r"""
extern "C" __global__
void rule30_batch_step(
    const unsigned long long* tapes,   // [n_variants * n_words]  read
    unsigned long long*       out,     // [n_variants * n_words]  write
    int n_variants,
    int n_words
) {
    int var = blockIdx.y;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (var >= n_variants || idx >= n_words) return;

    int base = var * n_words;
    unsigned long long center   = tapes[base + idx];
    unsigned long long prev_w   = (idx > 0)          ? tapes[base + idx - 1] : 0ULL;
    unsigned long long next_w   = (idx < n_words - 1) ? tapes[base + idx + 1] : 0ULL;

    unsigned long long left_w   = (center << 1) | (prev_w >> 63);
    unsigned long long right_w  = (center >> 1) | (next_w << 63);
    out[base + idx] = left_w ^ (center | right_w);
}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_initial_tape(n_cells: int) -> np.ndarray:
    """Single live cell at center, packed into uint64 words (LSB = leftmost)."""
    center_cell = n_cells // 2
    n_words = (n_cells + 63) // 64
    tape = np.zeros(n_words, dtype=np.uint64)
    word_idx  = center_cell // 64
    bit_idx   = center_cell %  64
    tape[word_idx] |= np.uint64(1) << np.uint64(bit_idx)
    return tape


def flip_bit(tape: np.ndarray, cell_pos: int) -> np.ndarray:
    """Return a copy of tape with bit at cell_pos flipped."""
    t = tape.copy()
    word_idx = cell_pos // 64
    bit_idx  = cell_pos %  64
    t[word_idx] ^= np.uint64(1) << np.uint64(bit_idx)
    return t


def extract_center_bit(tape_words: np.ndarray, n_cells: int) -> int:
    center = n_cells // 2
    w = int(tape_words[center // 64])
    return int((w >> (center % 64)) & 1)


# ---------------------------------------------------------------------------
# CPU fallback (no CuPy)
# ---------------------------------------------------------------------------

def rule30_step_cpu(tape: np.ndarray, n_words: int) -> np.ndarray:
    left  = (tape << np.uint64(1))
    left[1:] |= (tape[:-1] >> np.uint64(63))
    right = (tape >> np.uint64(1))
    right[:-1] |= (tape[1:] << np.uint64(63))
    return left ^ (tape | right)


def simulate_cpu(tape0: np.ndarray, n_steps: int, n_cells: int) -> np.ndarray:
    """Run n_steps on CPU; return center column as uint8 array."""
    tape = tape0.copy()
    n_words = len(tape)
    center_col = np.empty(n_steps, dtype=np.uint8)
    for t in range(n_steps):
        center_col[t] = extract_center_bit(tape, n_cells)
        tape = rule30_step_cpu(tape, n_words)
    return center_col


def step_naive(row: np.ndarray) -> np.ndarray:
    out = np.zeros_like(row)
    for i in range(len(row)):
        left = row[i - 1] if i > 0 else 0
        center = row[i]
        right = row[i + 1] if i + 1 < len(row) else 0
        out[i] = left ^ (center | right)
    return out


def verify_small_case() -> None:
    """Cheap preflight against a naive implementation on both sides of the center."""
    n_steps = 96
    n_cells = 2 * n_steps + 1
    center = n_cells // 2
    base = make_initial_tape(n_cells)
    ref = simulate_cpu(base, n_steps + 1, n_cells)

    naive_row = np.zeros(n_cells, dtype=np.uint8)
    naive_row[center] = 1
    naive_ref = np.empty(n_steps + 1, dtype=np.uint8)
    for t in range(n_steps + 1):
        naive_ref[t] = naive_row[center]
        naive_row = step_naive(naive_row)

    if not np.array_equal(ref, naive_ref):
        raise RuntimeError("Packed CPU kernel failed center-column verification against naive Rule 30.")

    for side_name, side_idx in [("LEFT", -1), ("RIGHT", +1)]:
        for dist in [1, 2, 3, 8, 9, 17, 31]:
            alt = flip_bit(base, center + side_idx * dist)
            col = simulate_cpu(alt, n_steps + 1, n_cells)
            diff = np.where(col != ref)[0]
            packed_first = int(diff[0]) if len(diff) else n_steps + 1

            row_a = np.zeros(n_cells, dtype=np.uint8)
            row_b = row_a.copy()
            row_a[center] = 1
            row_b[center] = 1
            row_b[center + side_idx * dist] ^= 1
            naive_first = n_steps + 1
            for t in range(n_steps + 1):
                if row_a[center] != row_b[center]:
                    naive_first = t
                    break
                row_a = step_naive(row_a)
                row_b = step_naive(row_b)

            if packed_first != naive_first:
                raise RuntimeError(
                    f"Packed CPU kernel failed perturbation check for {side_name} d={dist}: "
                    f"packed={packed_first}, naive={naive_first}."
                )


# ---------------------------------------------------------------------------
# GPU batch simulation
# ---------------------------------------------------------------------------

def simulate_batch_gpu(tapes_cpu: np.ndarray, n_steps: int, n_cells: int,
                       kernel) -> np.ndarray:
    """
    tapes_cpu: [n_variants, n_words] uint64
    Returns center_cols: [n_variants, n_steps] uint8

    Optimised: accumulate center bits on GPU -> single CPU sync per batch.
    """
    n_variants, n_words = tapes_cpu.shape
    center_word = (n_cells // 2) // 64
    center_bit  = (n_cells // 2) %  64

    tapes_a = cp.asarray(tapes_cpu)
    tapes_b = cp.zeros_like(tapes_a)
    # Accumulate center bits entirely on GPU (avoids n_steps CPU–GPU syncs)
    center_cols_gpu = cp.empty((n_variants, n_steps), dtype=cp.uint8)

    threads  = 128
    blocks_x = (n_words  + threads - 1) // threads
    blocks_y = n_variants

    for step in range(n_steps):
        center_cols_gpu[:, step] = (
            (tapes_a[:, center_word] >> cp.uint64(center_bit)) & cp.uint64(1)
        ).astype(cp.uint8)
        kernel(
            (blocks_x, blocks_y), (threads,),
            (tapes_a, tapes_b, np.int32(n_variants), np.int32(n_words))
        )
        tapes_a, tapes_b = tapes_b, tapes_a

    return cp.asnumpy(center_cols_gpu)   # single sync


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _log(msg):
    ts   = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    PROG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PROG_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    open(PROG_LOG, "w").close()   # clear progress log
    _log("=" * 65)
    _log(f"Experiment M — Causal Sensitivity Mapping  (TEST={TEST})")
    _log("=" * 65)
    _log(f"N_STEPS  = {N_STEPS:,}")
    _log(f"MAX_DIST = {MAX_DIST:,}  (flip distances 0..{MAX_DIST})")
    _log(f"N_CELLS  = {N_CELLS:,}  (tape width)")
    _log(f"BATCH    = {BATCH}")
    _log(f"GPU      = {GPU}")
    _log(f"Progress -> {PROG_LOG}")
    print()

    t0 = time.perf_counter()
    verify_small_case()
    _log("Preflight verification passed against naive Rule 30.")

    # Set up kernel
    if GPU:
        kernel = cp.RawKernel(KERNEL_SRC, "rule30_batch_step")
    n_words = (N_CELLS + 63) // 64

    # Original tape
    base_tape = make_initial_tape(N_CELLS)
    center_cell = N_CELLS // 2

    sim_len = N_STEPS + 1  # include time t = N_STEPS so distance N_STEPS is not censored

    # Simulate original (unflipped) run
    _log("Simulating unflipped reference run …")
    if GPU:
        orig_col = simulate_batch_gpu(
            base_tape.reshape(1, -1), sim_len, N_CELLS, kernel
        )[0]
    else:
        orig_col = simulate_cpu(base_tape, sim_len, N_CELLS)
    _log(f"  Reference done. fraction_ones={orig_col.mean():.4f}")

    # first_divergence[d, side] where side 0=left, 1=right
    # Value = first step where center column differs, or N_STEPS+1 if never within horizon
    first_div_left  = np.full(MAX_DIST + 1, sim_len, dtype=np.int32)
    first_div_right = np.full(MAX_DIST + 1, sim_len, dtype=np.int32)

    # Distances to test: 0 (flip center itself) up to MAX_DIST
    distances = list(range(MAX_DIST + 1))

    # Process in batches
    n_batches_total = 2 * ((MAX_DIST + BATCH) // BATCH)
    _log(f"\nRunning {2*(MAX_DIST+1):,} variants in {n_batches_total} batches of {BATCH} …")
    batch_num = 0
    for side_name, side_idx, results_arr in [
        ("LEFT",  -1, first_div_left),
        ("RIGHT", +1, first_div_right),
    ]:
        _log(f"  Side: {side_name}")
        for batch_start in tqdm(range(0, MAX_DIST + 1, BATCH), desc=f"  {side_name}"):
            batch_dists = distances[batch_start : batch_start + BATCH]
            n_var = len(batch_dists)

            tapes_batch = np.empty((n_var, n_words), dtype=np.uint64)
            for i, d in enumerate(batch_dists):
                flip_cell = center_cell + side_idx * d
                flip_cell = max(0, min(flip_cell, N_CELLS - 1))
                tapes_batch[i] = flip_bit(base_tape, flip_cell)

            if GPU:
                cols = simulate_batch_gpu(tapes_batch, sim_len, N_CELLS, kernel)
            else:
                cols = np.stack([simulate_cpu(tapes_batch[i], sim_len, N_CELLS)
                                 for i in range(n_var)])

            for i, d in enumerate(batch_dists):
                diff = (cols[i] != orig_col)
                first_steps = np.where(diff)[0]
                if len(first_steps) > 0:
                    results_arr[d] = int(first_steps[0])
                else:
                    results_arr[d] = sim_len  # never diverged in horizon 0..N_STEPS

            batch_num += 1
            if batch_num % 10 == 0:
                pct = 100.0 * batch_num / n_batches_total
                _log(f"    {pct:5.1f}%  batch {batch_num}/{n_batches_total}  "
                     f"dist {batch_start}–{min(batch_start+BATCH-1, MAX_DIST)}")

    elapsed = time.perf_counter() - t0
    _log(f"\nSimulation complete in {elapsed:.1f}s")

    # ---------------------------------------------------------------------------
    # Analysis
    # ---------------------------------------------------------------------------
    dists = np.arange(MAX_DIST + 1)

    never_diverged_left  = int(np.sum(first_div_left  == sim_len))
    never_diverged_right = int(np.sum(first_div_right == sim_len))
    boundary_speed_left  = int(np.sum(first_div_left  == dists))
    boundary_speed_right = int(np.sum(first_div_right == dists))
    causal_viol_left     = int(np.sum(first_div_left  < dists))
    causal_viol_right    = int(np.sum(first_div_right < dists))

    common_hit = (first_div_left <= N_STEPS) & (first_div_right <= N_STEPS)
    mean_asym = float(np.mean(np.abs(first_div_left[common_hit] - first_div_right[common_hit]))) if np.any(common_hit) else float("nan")

    print(f"\nResults:")
    print(f"  Never-diverged flips (left):  {never_diverged_left}/{MAX_DIST+1}")
    print(f"  Never-diverged flips (right): {never_diverged_right}/{MAX_DIST+1}")
    print(f"  Boundary-speed arrivals (left):  {boundary_speed_left}/{MAX_DIST+1}")
    print(f"  Boundary-speed arrivals (right): {boundary_speed_right}/{MAX_DIST+1}")
    print(f"  Causality violations (left/right): {causal_viol_left}/{causal_viol_right}")
    print(f"  Mean left-right asymmetry on common hits: {mean_asym:.2f} steps")

    # ---------------------------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------------------------
    result = {
        "n_steps":              N_STEPS,
        "sim_length":           sim_len,
        "max_dist":             MAX_DIST,
        "n_cells":              N_CELLS,
        "never_diverged_left":  never_diverged_left,
        "never_diverged_right": never_diverged_right,
        "boundary_speed_left":  boundary_speed_left,
        "boundary_speed_right": boundary_speed_right,
        "causal_violations_left": causal_viol_left,
        "causal_violations_right": causal_viol_right,
        "mean_asymmetry_steps": round(mean_asym, 3),
        "first_div_left":       first_div_left.tolist(),
        "first_div_right":      first_div_right.tolist(),
        "elapsed_s":            round(elapsed, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(str(OUT_JSON), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nJSON -> {OUT_JSON}")

    # ---------------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------------
    print("Generating plots …")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Experiment M — Causal Sensitivity Mapping\n"
                     f"Rule 30, {N_STEPS} steps, {MAX_DIST} flip distances",
                     fontsize=13, fontweight="bold")

        # Panel 1: first_divergence_step vs distance (scatter + light cone)
        ax = axes[0]
        ax.scatter(dists, first_div_left,  s=1, color="#2196F3", alpha=0.6, label="Left flip")
        ax.scatter(dists, first_div_right, s=1, color="#FF5722", alpha=0.6, label="Right flip")
        ax.plot(dists, dists, "k--", lw=1.2, label="Light cone T=D")
        ax.set_xlabel("Flip distance D from center")
        ax.set_ylabel("First step where center diverges")
        ax.set_title("First divergence step vs flip distance")
        ax.legend(markerscale=5, fontsize=9)
        ax.grid(True, alpha=0.3)

        # Panel 2: heatmap — did flip at distance D affect center by step T?
        # Build binary matrix [n_dists x n_steps] (subsample for display)
        step_sample  = max(1, N_STEPS  // 300)
        dist_sample  = max(1, MAX_DIST // 300)
        sampled_dists = dists[::dist_sample]
        sampled_steps = np.arange(0, N_STEPS, step_sample)
        mat = np.zeros((len(sampled_dists), len(sampled_steps)), dtype=np.float32)
        for i, d in enumerate(sampled_dists):
            for j, t in enumerate(sampled_steps):
                # 1 if center has diverged by step t for this flip
                mat[i, j] = float(first_div_left[d] <= t)

        ax = axes[1]
        im = ax.imshow(mat, aspect="auto", origin="lower",
                       extent=[0, N_STEPS, 0, MAX_DIST],
                       cmap="Blues", interpolation="nearest")
        ax.plot(dists[::dist_sample], dists[::dist_sample], "r--", lw=1.2,
                label="Light cone")
        ax.set_xlabel("Step T")
        ax.set_ylabel("Flip distance D (left side)")
        ax.set_title("Causal cone heatmap (left flips)\nBlue = center affected by step T")
        ax.legend(fontsize=9)
        plt.colorbar(im, ax=ax, label="Diverged?")

        # Panel 3: Asymmetry — left vs right first-divergence step
        ax = axes[2]
        ax.scatter(first_div_left, first_div_right, s=1, color="#9C27B0", alpha=0.5)
        lim_max = N_STEPS
        ax.plot([0, lim_max], [0, lim_max], "k--", lw=1, label="Symmetry line")
        ax.set_xlabel("First divergence step (left flip)")
        ax.set_ylabel("First divergence step (right flip)")
        ax.set_title(f"Left vs right symmetry\nmean asymmetry={mean_asym:.1f} steps")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)

        plt.tight_layout()
        PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(PLOT_FILE), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot -> {PLOT_FILE}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    # ---------------------------------------------------------------------------
    # Experiment log
    # ---------------------------------------------------------------------------
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(str(LOG_FILE), "w", encoding="utf-8") as f:
        f.write(f"""# Experiment Log

- Date: {datetime.date.today().isoformat()}
- Title: Causal Sensitivity Mapping
- Goal: Map how single-bit perturbations propagate to the center column — tests whether Rule 30's sensitivity is uniform, asymmetric, or has blind spots
- Setup: N_STEPS={N_STEPS}, MAX_DIST={MAX_DIST}, N_CELLS={N_CELLS}, BATCH={BATCH}, GPU={GPU}
- Method: For each flip distance D (left and right of center), flip that initial bit and measure the first step at which the center column diverges from the unflipped run
- Result:
  - Never-diverged (left):  {never_diverged_left}/{MAX_DIST+1}
  - Never-diverged (right): {never_diverged_right}/{MAX_DIST+1}
  - Boundary-speed arrivals T=D (left):  {boundary_speed_left}/{MAX_DIST+1}
  - Boundary-speed arrivals T=D (right): {boundary_speed_right}/{MAX_DIST+1}
  - Causality violations T<D (left/right): {causal_viol_left}/{causal_viol_right}
  - Mean left-right asymmetry on common hits: {mean_asym:.2f} steps
- Interpretation: {"No causality violations detected after packed-bit fix." if (causal_viol_left + causal_viol_right) == 0 else "Causality violations remain — implementation still needs inspection."}
  {"Both sides reach the center equally quickly." if mean_asym < 50 else "Strong left-right asymmetry remains after the implementation fix."}
- Verification: Plot saved to docs/plots/. Three panels: (1) first-divergence scatter vs light cone, (2) causal cone heatmap, (3) left vs right symmetry.
- Elapsed: {elapsed:.0f}s
""")
    print(f"Log  -> {LOG_FILE}")
    print(f"\nDone. Total: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

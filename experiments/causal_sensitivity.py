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
N_STEPS   = 3000     # simulate this many steps per variant
MAX_DIST  = 2000     # flip distances 0 .. MAX_DIST from center
# Tape must be at least 2*N_STEPS + 1 wide to prevent edge effects
N_CELLS   = 2 * N_STEPS + 1   # = 6001 — minimal safe width
BATCH     = 100      # process this many flip distances at once on GPU

OUT_JSON  = Path(r"D:\APATPROJECTS\rule30-research\data\causal_sensitivity.json")
PLOT_FILE = Path(r"D:\APATPROJECTS\rule30-research\docs\plots\causal_sensitivity.png")
LOG_FILE  = Path(r"D:\APATPROJECTS\rule30-research\docs\experiment-logs\M_causal_sensitivity.md")

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

    unsigned long long left_w   = (center >> 1) | (prev_w << 63);
    unsigned long long right_w  = (center << 1) | (next_w >> 63);
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
    left  = np.zeros(n_words, dtype=np.uint64)
    right = np.zeros(n_words, dtype=np.uint64)
    left[1:]  = (tape[:-1] << np.uint64(63)) | (tape[1:]  >> np.uint64(1))
    right[:-1]= (tape[1:]  << np.uint64(1))  | (tape[:-1] >> np.uint64(63))
    # left of center word: (tape >> 1) | (prev << 63) — done above
    left  = (tape >> np.uint64(1))
    left[1:] |= (tape[:-1] << np.uint64(63))
    right = (tape << np.uint64(1))
    right[:-1] |= (tape[1:] >> np.uint64(63))
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


# ---------------------------------------------------------------------------
# GPU batch simulation
# ---------------------------------------------------------------------------

def simulate_batch_gpu(tapes_cpu: np.ndarray, n_steps: int, n_cells: int,
                       kernel) -> np.ndarray:
    """
    tapes_cpu: [n_variants, n_words] uint64
    Returns center_cols: [n_variants, n_steps] uint8
    """
    n_variants, n_words = tapes_cpu.shape
    center_word = (n_cells // 2) // 64
    center_bit  = (n_cells // 2) %  64

    tapes_a = cp.asarray(tapes_cpu)   # ping
    tapes_b = cp.zeros_like(tapes_a)  # pong

    center_cols = np.empty((n_variants, n_steps), dtype=np.uint8)

    threads = 128
    blocks_x = (n_words     + threads - 1) // threads
    blocks_y = n_variants

    for step in range(n_steps):
        # Extract center bits for all variants
        col_words = cp.asnumpy(tapes_a[:, center_word])
        for v in range(n_variants):
            center_cols[v, step] = int((int(col_words[v]) >> center_bit) & 1)

        kernel(
            (blocks_x, blocks_y), (threads,),
            (tapes_a, tapes_b, np.int32(n_variants), np.int32(n_words))
        )
        tapes_a, tapes_b = tapes_b, tapes_a

    return center_cols


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Experiment M — Causal Sensitivity Mapping")
    print("=" * 65)
    print(f"N_STEPS  = {N_STEPS:,}")
    print(f"MAX_DIST = {MAX_DIST:,}  (flip distances 0..{MAX_DIST})")
    print(f"N_CELLS  = {N_CELLS:,}  (tape width)")
    print(f"BATCH    = {BATCH}")
    print(f"GPU      = {GPU}")
    print()

    t0 = time.perf_counter()

    # Set up kernel
    if GPU:
        kernel = cp.RawKernel(KERNEL_SRC, "rule30_batch_step")
    n_words = (N_CELLS + 63) // 64

    # Original tape
    base_tape = make_initial_tape(N_CELLS)
    center_cell = N_CELLS // 2

    # Simulate original (unflipped) run
    print("Simulating unflipped reference run …")
    if GPU:
        orig_col = simulate_batch_gpu(
            base_tape.reshape(1, -1), N_STEPS, N_CELLS, kernel
        )[0]
    else:
        orig_col = simulate_cpu(base_tape, N_STEPS, N_CELLS)
    print(f"  Reference done. fraction_ones={orig_col.mean():.4f}")

    # first_divergence[d, side] where side 0=left, 1=right
    # Value = first step where center column differs, or N_STEPS if never
    first_div_left  = np.full(MAX_DIST + 1, N_STEPS, dtype=np.int32)
    first_div_right = np.full(MAX_DIST + 1, N_STEPS, dtype=np.int32)

    # Distances to test: 0 (flip center itself) up to MAX_DIST
    distances = list(range(MAX_DIST + 1))

    # Process in batches
    print(f"\nRunning {2 * (MAX_DIST + 1):,} variants in batches of {BATCH} …")
    for side_name, side_idx, results_arr in [
        ("LEFT",  -1, first_div_left),
        ("RIGHT", +1, first_div_right),
    ]:
        print(f"  Side: {side_name}")
        for batch_start in tqdm(range(0, MAX_DIST + 1, BATCH), desc=f"  {side_name}"):
            batch_dists = distances[batch_start : batch_start + BATCH]
            n_var = len(batch_dists)

            tapes_batch = np.empty((n_var, n_words), dtype=np.uint64)
            for i, d in enumerate(batch_dists):
                flip_cell = center_cell + side_idx * d
                flip_cell = max(0, min(flip_cell, N_CELLS - 1))
                tapes_batch[i] = flip_bit(base_tape, flip_cell)

            if GPU:
                cols = simulate_batch_gpu(tapes_batch, N_STEPS, N_CELLS, kernel)
            else:
                cols = np.stack([simulate_cpu(tapes_batch[i], N_STEPS, N_CELLS)
                                 for i in range(n_var)])

            for i, d in enumerate(batch_dists):
                diff = (cols[i] != orig_col)
                first_steps = np.where(diff)[0]
                if len(first_steps) > 0:
                    results_arr[d] = int(first_steps[0])
                else:
                    results_arr[d] = N_STEPS  # never diverged in N_STEPS

    elapsed = time.perf_counter() - t0
    print(f"\nSimulation complete in {elapsed:.1f}s")

    # ---------------------------------------------------------------------------
    # Analysis
    # ---------------------------------------------------------------------------
    dists = np.arange(MAX_DIST + 1)

    # Light-cone boundary: first_div should be <= D for all D (if deterministic chaos)
    never_diverged_left  = int(np.sum(first_div_left  == N_STEPS))
    never_diverged_right = int(np.sum(first_div_right == N_STEPS))

    # How many hit the light cone exactly (first_div[d] <= d)?
    on_cone_left  = int(np.sum(first_div_left  <= dists))
    on_cone_right = int(np.sum(first_div_right <= dists))

    # Asymmetry: mean |left_step - right_step| for same distance
    asym = np.abs(first_div_left.astype(float) - first_div_right.astype(float))
    mean_asym = float(asym.mean())

    print(f"\nResults:")
    print(f"  Never-diverged flips (left):  {never_diverged_left}/{MAX_DIST+1}")
    print(f"  Never-diverged flips (right): {never_diverged_right}/{MAX_DIST+1}")
    print(f"  Flips on/within light cone (left):  {on_cone_left}/{MAX_DIST+1}")
    print(f"  Flips on/within light cone (right): {on_cone_right}/{MAX_DIST+1}")
    print(f"  Mean left-right asymmetry: {mean_asym:.2f} steps")

    # ---------------------------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------------------------
    result = {
        "n_steps":              N_STEPS,
        "max_dist":             MAX_DIST,
        "n_cells":              N_CELLS,
        "never_diverged_left":  never_diverged_left,
        "never_diverged_right": never_diverged_right,
        "on_cone_left":         on_cone_left,
        "on_cone_right":        on_cone_right,
        "mean_asymmetry_steps": round(mean_asym, 3),
        "first_div_left":       first_div_left.tolist(),
        "first_div_right":      first_div_right.tolist(),
        "elapsed_s":            round(elapsed, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(str(OUT_JSON), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nJSON → {OUT_JSON}")

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
        print(f"Plot → {PLOT_FILE}")
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
  - Flips within light cone (T<=D), left:  {on_cone_left}/{MAX_DIST+1}
  - Flips within light cone (T<=D), right: {on_cone_right}/{MAX_DIST+1}
  - Mean left-right asymmetry: {mean_asym:.2f} steps
- Interpretation: {"All flips eventually reached center — full sensitivity. Light cone boundary T=D holds." if never_diverged_left + never_diverged_right == 0 else f"Some flips did NOT affect center in {N_STEPS} steps — potential insensitive directions."}
  {"Symmetric propagation: left/right asymmetry is small." if mean_asym < 50 else "Asymmetric propagation detected — Rule 30 favors one direction."}
- Verification: Plot saved to docs/plots/. Three panels: (1) first-divergence scatter vs light cone, (2) causal cone heatmap, (3) left vs right symmetry.
- Elapsed: {elapsed:.0f}s
""")
    print(f"Log  → {LOG_FILE}")
    print(f"\nDone. Total: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

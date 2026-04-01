#!/usr/bin/env python
"""Transfer entropy along Rule 30 causal diagonals."""

from __future__ import annotations

import json
import time
import datetime
import sys
from pathlib import Path

import numpy as np

try:
    import cupy as cp

    try:
        GPU = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        GPU = False
except ImportError:
    cp = None
    GPU = False

from rule30_open_utils import verify_spacetime_against_naive


ROOT = Path(r"D:\APATPROJECTS\rule30-research")
OUT_JSON = ROOT / "data" / "diagonal_te.json"
PLOT_FILE = ROOT / "docs" / "plots" / "diagonal_te.png"
PROG_LOG = ROOT / "data" / "diagonal_te.progress.log"

TEST = "--test" in sys.argv
N_SIM_STEPS = 20_000 if TEST else 400_000
STRIP_WIDTH = 64 if TEST else 512
CHUNK_SIZE = 2_000 if TEST else 10_000
DISTANCES = [1, 2, 4, 8, 16, 32] if TEST else [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
SURROGATE_SHIFT = 257 if TEST else 9973
BURN_IN = max(DISTANCES) + 1024

KERNEL_SRC = r"""
extern "C" __global__
void rule30_step(
    const unsigned long long* tape,
    unsigned long long* out,
    int n_words
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_words) return;
    unsigned long long c = tape[idx];
    unsigned long long prev = (idx > 0) ? tape[idx - 1] : 0ULL;
    unsigned long long next = (idx < n_words - 1) ? tape[idx + 1] : 0ULL;
    unsigned long long left = (c << 1) | (prev >> 63);
    unsigned long long right = (c >> 1) | (next << 63);
    out[idx] = left ^ (c | right);
}
"""


def log(msg: str) -> None:
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with open(PROG_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def te_from_counts(counts8: np.ndarray) -> float:
    total = int(counts8.sum())
    if total == 0:
        return 0.0
    p = counts8.astype(np.float64) / total
    cube = p.reshape(2, 2, 2)

    def h(arr: np.ndarray) -> float:
        flat = arr.flatten()
        flat = flat[flat > 1e-15]
        return float(-np.sum(flat * np.log2(flat)))

    p_yhist_y = cube.sum(axis=0)
    p_yhist = p_yhist_y.sum(axis=1)
    p_x_yhist = cube.sum(axis=2)
    return max(0.0, (h(p_yhist_y) - h(p_yhist)) - (h(cube) - h(p_x_yhist)))


def make_initial_tape(n_cells: int, center_cell: int) -> np.ndarray:
    n_words = (n_cells + 63) // 64
    tape = np.zeros(n_words, dtype=np.uint64)
    tape[center_cell // 64] |= np.uint64(1) << np.uint64(center_cell % 64)
    return tape


def cpu_step(tape: np.ndarray) -> np.ndarray:
    left = tape << np.uint64(1)
    left[1:] |= tape[:-1] >> np.uint64(63)
    right = tape >> np.uint64(1)
    right[:-1] |= tape[1:] << np.uint64(63)
    return left ^ (tape | right)


def simulate_strip_chunks():
    n_cells = 2 * N_SIM_STEPS + 1
    n_words = (n_cells + 63) // 64
    center = n_cells // 2
    strip_left = center - STRIP_WIDTH
    strip_right = center + STRIP_WIDTH
    col_positions = np.arange(strip_left, strip_right + 1)
    local_words = (col_positions // 64).astype(np.intp)
    bit_shifts = (col_positions % 64).astype(np.uint64)

    tape_a = make_initial_tape(n_cells, center)
    if GPU:
        kernel = cp.RawKernel(KERNEL_SRC, "rule30_step")
        cur = cp.asarray(tape_a)
        nxt = cp.zeros_like(cur)
        strip_buf = cp.zeros((CHUNK_SIZE, len(local_words)), dtype=cp.uint64)
        word_idx = cp.asarray(local_words, dtype=cp.int32)
        threads = 256
        blocks = (n_words + threads - 1) // threads
    else:
        cur = tape_a.copy()
        strip_buf_cpu = np.zeros((CHUNK_SIZE, len(local_words)), dtype=np.uint64)

    n_chunks = (N_SIM_STEPS + CHUNK_SIZE - 1) // CHUNK_SIZE
    for chunk_idx in range(n_chunks):
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, N_SIM_STEPS)
        chunk_len = end - start
        if GPU:
            for step in range(chunk_len):
                strip_buf[step] = cur[word_idx]
                kernel((blocks,), (threads,), (cur, nxt, np.int32(n_words)))
                cur, nxt = nxt, cur
            packed = cp.asnumpy(strip_buf[:chunk_len])
        else:
            for step in range(chunk_len):
                strip_buf_cpu[step] = cur[local_words]
                cur = cpu_step(cur)
            packed = strip_buf_cpu[:chunk_len].copy()

        bits = ((packed[:, np.arange(len(local_words))] >> bit_shifts) & 1).astype(np.uint8)
        yield bits
        if chunk_idx == 0 or (chunk_idx + 1) % 10 == 0 or chunk_idx == n_chunks - 1:
            log(f"  chunk {chunk_idx + 1}/{n_chunks} ready")


def main() -> None:
    PROG_LOG.parent.mkdir(parents=True, exist_ok=True)
    open(PROG_LOG, "w").close()
    verify_spacetime_against_naive()
    log("Packed spacetime strip extraction verified against naive Rule 30.")

    t0 = time.perf_counter()
    log(
        f"Simulating diagonal TE with N_SIM_STEPS={N_SIM_STEPS:,}, strip_width={STRIP_WIDTH}, "
        f"distances={DISTANCES}, GPU={GPU}"
    )

    center_idx = STRIP_WIDTH
    max_d = max(DISTANCES)
    carry = np.zeros((max_d + 1, 2 * STRIP_WIDTH + 1), dtype=np.uint8)

    diag_counts_left = {d: [] for d in DISTANCES}
    diag_counts_right = {d: [] for d in DISTANCES}
    target_store = {d: [] for d in DISTANCES}
    history_store = {d: [] for d in DISTANCES}

    step_offset = 0
    for bits in simulate_strip_chunks():
        window = np.concatenate([carry, bits], axis=0)
        chunk_len = len(bits)
        current_center = bits[:, center_idx]
        carry_len = max_d + 1
        prev_center = window[carry_len - 1:carry_len - 1 + chunk_len, center_idx]
        global_times = step_offset + np.arange(chunk_len)

        for d in DISTANCES:
            source_left = window[carry_len - d:carry_len - d + chunk_len, center_idx - d]
            source_right = window[carry_len - d:carry_len - d + chunk_len, center_idx + d]
            valid = (global_times >= BURN_IN) & (global_times >= d) & (global_times >= 1)
            if not np.any(valid):
                continue
            target_store[d].append(current_center[valid].copy())
            history_store[d].append(prev_center[valid].copy())
            diag_counts_left[d].append(source_left[valid].copy())
            diag_counts_right[d].append(source_right[valid].copy())

        carry = window[-(max_d + 1):].copy()
        step_offset += chunk_len

    results: dict[str, object] = {
        "date": datetime.date.today().isoformat(),
        "n_sim_steps": N_SIM_STEPS,
        "strip_width": STRIP_WIDTH,
        "burn_in": BURN_IN,
        "surrogate_shift": SURROGATE_SHIFT,
        "distances": DISTANCES,
        "gpu": GPU,
        "samples": {},
        "te_left_diag": [],
        "te_right_diag": [],
        "te_left_diag_surrogate": [],
        "te_right_diag_surrogate": [],
        "te_left_diag_excess": [],
        "te_right_diag_excess": [],
    }

    for d in DISTANCES:
        y = np.concatenate(target_store[d]).astype(np.int32)
        y_prev = np.concatenate(history_store[d]).astype(np.int32)
        x_left = np.concatenate(diag_counts_left[d]).astype(np.int32)
        x_right = np.concatenate(diag_counts_right[d]).astype(np.int32)
        shift = SURROGATE_SHIFT % len(y)
        if shift == 0:
            shift = 1
        x_left_s = np.roll(x_left, shift)
        x_right_s = np.roll(x_right, shift)

        counts_left = np.bincount(y + 2 * y_prev + 4 * x_left, minlength=8)
        counts_right = np.bincount(y + 2 * y_prev + 4 * x_right, minlength=8)
        counts_left_s = np.bincount(y + 2 * y_prev + 4 * x_left_s, minlength=8)
        counts_right_s = np.bincount(y + 2 * y_prev + 4 * x_right_s, minlength=8)

        te_left = te_from_counts(counts_left)
        te_right = te_from_counts(counts_right)
        te_left_s = te_from_counts(counts_left_s)
        te_right_s = te_from_counts(counts_right_s)
        results["samples"][str(d)] = int(len(y))
        results["te_left_diag"].append(round(te_left, 8))
        results["te_right_diag"].append(round(te_right, 8))
        results["te_left_diag_surrogate"].append(round(te_left_s, 8))
        results["te_right_diag_surrogate"].append(round(te_right_s, 8))
        results["te_left_diag_excess"].append(round(te_left - te_left_s, 8))
        results["te_right_diag_excess"].append(round(te_right - te_right_s, 8))
        log(
            f"d={d:>4}: left={te_left:.6f} (surr {te_left_s:.6f})  "
            f"right={te_right:.6f} (surr {te_right_s:.6f})"
        )

    elapsed = time.perf_counter() - t0
    results["elapsed_s"] = round(elapsed, 2)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"JSON -> {OUT_JSON}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ds = np.asarray(DISTANCES)
        left = np.asarray(results["te_left_diag"], dtype=np.float64)
        right = np.asarray(results["te_right_diag"], dtype=np.float64)
        left_s = np.asarray(results["te_left_diag_surrogate"], dtype=np.float64)
        right_s = np.asarray(results["te_right_diag_surrogate"], dtype=np.float64)
        left_ex = np.asarray(results["te_left_diag_excess"], dtype=np.float64)
        right_ex = np.asarray(results["te_right_diag_excess"], dtype=np.float64)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            "Diagonal transfer entropy in Rule 30\n"
            f"{N_SIM_STEPS:,} simulated steps, burn-in {BURN_IN}",
            fontsize=12,
            fontweight="bold",
        )

        ax = axes[0]
        ax.semilogx(ds, left, "o-", label="Left diagonal")
        ax.semilogx(ds, right, "s--", label="Right diagonal")
        ax.semilogx(ds, left_s, ":", color="steelblue", label="Left surrogate")
        ax.semilogx(ds, right_s, ":", color="tomato", label="Right surrogate")
        ax.set_xlabel("Diagonal distance")
        ax.set_ylabel("TE (bits)")
        ax.set_title("Raw TE vs surrogate")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1]
        ax.semilogx(ds, left_ex, "o-", label="Left excess")
        ax.semilogx(ds, right_ex, "s--", label="Right excess")
        ax.axhline(0.0, color="black", lw=1)
        ax.set_xlabel("Diagonal distance")
        ax.set_ylabel("TE above surrogate")
        ax.set_title("Excess TE along causal diagonals")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[2]
        ax.bar(np.arange(len(ds)), left_ex - right_ex, color="purple")
        ax.axhline(0.0, color="black", lw=1)
        ax.set_xticks(np.arange(len(ds)))
        ax.set_xticklabels([str(d) for d in ds], rotation=45)
        ax.set_xlabel("Diagonal distance")
        ax.set_ylabel("Left excess - right excess")
        ax.set_title("Directional asymmetry")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
        plt.close()
        log(f"Plot -> {PLOT_FILE}")
    except Exception as exc:
        log(f"Plot skipped: {exc}")


if __name__ == "__main__":
    main()

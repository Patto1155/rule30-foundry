#!/usr/bin/env python
"""Repeat causal sensitivity from multiple random initial conditions."""

from __future__ import annotations

import json
import time
import datetime
import sys
from pathlib import Path

import numpy as np

from rule30_open_utils import (
    GPU_AVAILABLE,
    first_divergence_steps,
    simulate_center_columns_batch,
    verify_random_batch_against_naive,
    verify_single_spike_direction_and_boundary,
)


ROOT = Path(r"D:\APATPROJECTS\rule30-research")
OUT_JSON = ROOT / "data" / "causal_sensitivity_multiseed.json"
PLOT_FILE = ROOT / "docs" / "plots" / "causal_sensitivity_multiseed.png"
PROG_LOG = ROOT / "data" / "causal_sensitivity_multiseed.progress.log"

TEST = "--test" in sys.argv
N_STEPS = 256 if TEST else 4096
MAX_DIST = 256 if TEST else 4096
N_SEEDS = 3 if TEST else 8
BATCH = 64 if TEST else 512


def log(msg: str) -> None:
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with open(PROG_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def summarize_seed(distances: np.ndarray, left: np.ndarray, right: np.ndarray, horizon: int, seed: int) -> dict[str, object]:
    left_never = int(np.sum(left == horizon + 1))
    right_never = int(np.sum(right == horizon + 1))
    left_boundary = int(np.sum(left == distances))
    right_boundary = int(np.sum(right == distances))
    common = (left <= horizon) & (right <= horizon)
    asym_values = left[common] - right[common]
    return {
        "seed": seed,
        "left_never": left_never,
        "right_never": right_never,
        "left_boundary": left_boundary,
        "right_boundary": right_boundary,
        "causal_violations_left": int(np.sum(left < distances)),
        "causal_violations_right": int(np.sum(right < distances)),
        "mean_signed_asymmetry": round(float(asym_values.mean()) if asym_values.size else 0.0, 6),
        "mean_abs_asymmetry": round(float(np.abs(asym_values).mean()) if asym_values.size else 0.0, 6),
        "right_minus_left_never": right_never - left_never,
    }


def main() -> None:
    PROG_LOG.parent.mkdir(parents=True, exist_ok=True)
    open(PROG_LOG, "w").close()
    verify_single_spike_direction_and_boundary()
    verify_random_batch_against_naive()
    log("Packed open-boundary verification passed on single-spike and random rows.")

    n_cells = 2 * N_STEPS + 1
    center = N_STEPS
    distances = np.arange(MAX_DIST + 1, dtype=np.int32)
    rng = np.random.default_rng(20260401)

    seed_summaries: list[dict[str, object]] = []
    left_never_counts: list[int] = []
    right_never_counts: list[int] = []
    mean_signed_asym: list[float] = []

    t0 = time.perf_counter()
    for seed_idx in range(N_SEEDS):
        row_seed = int(rng.integers(0, 2**31 - 1))
        row_rng = np.random.default_rng(row_seed)
        base = row_rng.integers(0, 2, size=n_cells, dtype=np.uint8)
        reference = simulate_center_columns_batch(base, N_STEPS, center, gpu=GPU_AVAILABLE)[0]

        left_first = np.full(MAX_DIST + 1, N_STEPS + 1, dtype=np.int32)
        right_first = np.full(MAX_DIST + 1, N_STEPS + 1, dtype=np.int32)

        for side_name, side_sign, out in [("left", -1, left_first), ("right", 1, right_first)]:
            for start in range(0, MAX_DIST + 1, BATCH):
                batch_dists = distances[start:start + BATCH]
                rows = np.repeat(base[None, :], len(batch_dists), axis=0)
                rows[np.arange(len(batch_dists)), center + side_sign * batch_dists] ^= 1
                cols = simulate_center_columns_batch(rows, N_STEPS, center, gpu=GPU_AVAILABLE)
                out[start:start + len(batch_dists)] = first_divergence_steps(reference, cols, N_STEPS + 1)
            impossible = out < distances
            if np.any(impossible):
                bad = int(np.flatnonzero(impossible)[0])
                raise RuntimeError(f"Causality violation for seed {seed_idx} on {side_name} side at d={bad}.")

        summary = summarize_seed(distances, left_first, right_first, N_STEPS, row_seed)
        summary["first_div_left"] = left_first.tolist()
        summary["first_div_right"] = right_first.tolist()
        seed_summaries.append(summary)
        left_never_counts.append(summary["left_never"])
        right_never_counts.append(summary["right_never"])
        mean_signed_asym.append(float(summary["mean_signed_asymmetry"]))
        log(
            f"seed {seed_idx + 1}/{N_SEEDS} ({row_seed}): "
            f"left_never={summary['left_never']} right_never={summary['right_never']} "
            f"mean_signed_asym={summary['mean_signed_asymmetry']:+.2f}"
        )

    elapsed = time.perf_counter() - t0

    result = {
        "date": datetime.date.today().isoformat(),
        "n_steps": N_STEPS,
        "max_dist": MAX_DIST,
        "n_cells": n_cells,
        "n_seeds": N_SEEDS,
        "gpu": GPU_AVAILABLE,
        "batch": BATCH,
        "seed_summaries": seed_summaries,
        "left_never_mean": round(float(np.mean(left_never_counts)), 6),
        "right_never_mean": round(float(np.mean(right_never_counts)), 6),
        "left_never_std": round(float(np.std(left_never_counts)), 6),
        "right_never_std": round(float(np.std(right_never_counts)), 6),
        "mean_signed_asymmetry": round(float(np.mean(mean_signed_asym)), 6),
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

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            "Multi-seed causal sensitivity\n"
            f"{N_SEEDS} random initial rows, horizon {N_STEPS}",
            fontsize=12,
            fontweight="bold",
        )

        ax = axes[0]
        seed_ids = np.arange(1, N_SEEDS + 1)
        ax.plot(seed_ids, left_never_counts, "o-", label="Left censored count")
        ax.plot(seed_ids, right_never_counts, "s--", label="Right censored count")
        ax.set_xlabel("Seed index")
        ax.set_ylabel("Count censored by horizon")
        ax.set_title("Non-arrivals by side")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1]
        ax.axhline(0.0, color="black", lw=1)
        ax.bar(seed_ids, mean_signed_asym, color=["tomato" if x < 0 else "steelblue" for x in mean_signed_asym])
        ax.set_xlabel("Seed index")
        ax.set_ylabel("Mean(left arrival - right arrival)")
        ax.set_title("Signed asymmetry on common hits")
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

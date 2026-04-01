#!/usr/bin/env python
"""Extend the horizon for right-side causal non-arrivals only."""

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
    make_single_spike_row,
    simulate_center_columns_batch,
    verify_single_spike_direction_and_boundary,
)


ROOT = Path(r"D:\APATPROJECTS\rule30-research")
IN_JSON = ROOT / "data" / "causal_sensitivity.json"
OUT_JSON = ROOT / "data" / "right_nonarrival_persistence.json"
PLOT_FILE = ROOT / "docs" / "plots" / "right_nonarrival_persistence.png"
PROG_LOG = ROOT / "data" / "right_nonarrival_persistence.progress.log"

TEST = "--test" in sys.argv
EXTENDED_STEPS = 15_000 if TEST else 50_000
BATCH = 64 if TEST else 256


def log(msg: str) -> None:
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with open(PROG_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    PROG_LOG.parent.mkdir(parents=True, exist_ok=True)
    open(PROG_LOG, "w").close()

    with open(IN_JSON, "r", encoding="utf-8") as f:
        prior = json.load(f)

    prior_horizon = int(prior["n_steps"])
    prior_sim_length = int(prior["sim_length"])
    prior_right = np.asarray(prior["first_div_right"], dtype=np.int32)
    blind = np.flatnonzero(prior_right == prior_sim_length).astype(np.int32)
    if TEST:
        blind = blind[:128]
    if blind.size == 0:
        raise RuntimeError("No right-side non-arrivals were found in data/causal_sensitivity.json.")
    if EXTENDED_STEPS <= prior_horizon:
        raise RuntimeError("Extended horizon must exceed the original 10,000-step horizon.")

    verify_single_spike_direction_and_boundary()
    log("Packed open-boundary verification passed.")

    max_dist = int(blind.max())
    center = EXTENDED_STEPS
    n_cells = 2 * EXTENDED_STEPS + max_dist + 1
    base = make_single_spike_row(n_cells, center)
    reference = simulate_center_columns_batch(base, EXTENDED_STEPS, center, gpu=GPU_AVAILABLE)[0]
    if reference.shape[0] != EXTENDED_STEPS + 1:
        raise RuntimeError("Reference center-column length mismatch.")

    rerun_first = np.full(len(blind), EXTENDED_STEPS + 1, dtype=np.int32)
    t0 = time.perf_counter()
    n_batches = (len(blind) + BATCH - 1) // BATCH
    log(
        f"Rerunning {len(blind)} right-side censored distances to {EXTENDED_STEPS:,} steps "
        f"with n_cells={n_cells:,}, batch={BATCH}, GPU={GPU_AVAILABLE}"
    )

    for batch_idx, start in enumerate(range(0, len(blind), BATCH), start=1):
        batch_dists = blind[start:start + BATCH]
        rows = np.repeat(base[None, :], len(batch_dists), axis=0)
        rows[np.arange(len(batch_dists)), center + batch_dists] ^= 1
        cols = simulate_center_columns_batch(rows, EXTENDED_STEPS, center, gpu=GPU_AVAILABLE)
        first = first_divergence_steps(reference, cols, EXTENDED_STEPS + 1)
        impossible = first < batch_dists
        if np.any(impossible):
            bad = int(batch_dists[np.flatnonzero(impossible)[0]])
            raise RuntimeError(f"Causality violation detected in persistence rerun at distance {bad}.")
        rerun_first[start:start + len(batch_dists)] = first
        if batch_idx == 1 or batch_idx % 4 == 0 or batch_idx == n_batches:
            log(f"  batch {batch_idx}/{n_batches} complete")

    elapsed = time.perf_counter() - t0

    slow_arrivals_mask = (rerun_first > prior_horizon) & (rerun_first <= EXTENDED_STEPS)
    still_censored_mask = rerun_first == EXTENDED_STEPS + 1
    slow_dists = blind[slow_arrivals_mask]
    censored_dists = blind[still_censored_mask]

    result = {
        "date": datetime.date.today().isoformat(),
        "prior_horizon": prior_horizon,
        "extended_horizon": EXTENDED_STEPS,
        "gpu": GPU_AVAILABLE,
        "batch": BATCH,
        "n_cells": n_cells,
        "rerun_distances": blind.tolist(),
        "rerun_first_divergence": rerun_first.tolist(),
        "new_arrivals_count": int(np.sum(slow_arrivals_mask)),
        "still_censored_count": int(np.sum(still_censored_mask)),
        "new_arrivals_examples": slow_dists[:25].tolist(),
        "still_censored_examples": censored_dists[:25].tolist(),
        "latest_new_arrivals": slow_dists[-25:].tolist() if slow_dists.size else [],
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

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            "Persistence of right-side non-arrivals\n"
            f"original horizon 10,000, rerun horizon {EXTENDED_STEPS:,}",
            fontsize=12,
            fontweight="bold",
        )

        ax = axes[0]
        ax.scatter(blind, rerun_first, s=8, color="black", alpha=0.6)
        ax.axhline(prior_horizon, color="tomato", ls="--", label="Original horizon")
        ax.axhline(EXTENDED_STEPS, color="steelblue", ls=":", label="Extended horizon")
        ax.set_xlabel("Right-side distance")
        ax.set_ylabel("First divergence step after rerun")
        ax.set_title("Targeted rerun outcomes")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1]
        categories = ["Arrived by 50k", "Still censored at 50k"]
        counts = [int(np.sum(slow_arrivals_mask)), int(np.sum(still_censored_mask))]
        ax.bar(categories, counts, color=["darkgreen", "tomato"])
        ax.set_ylabel("Count")
        ax.set_title("Outcome split for prior non-arrivals")
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

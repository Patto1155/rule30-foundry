#!/usr/bin/env python
"""Directional correlation spectrum across a richer family of slopes."""

from __future__ import annotations

import json
import time
import datetime
import sys
from pathlib import Path

import numpy as np

from rule30_open_utils import GPU_AVAILABLE, make_single_spike_row, simulate_spacetime, verify_spacetime_against_naive


ROOT = Path(r"D:\APATPROJECTS\rule30-research")
OUT_JSON = ROOT / "data" / "anisotropy_spectrum.json"
PLOT_FILE = ROOT / "docs" / "plots" / "anisotropy_spectrum.png"
PROG_LOG = ROOT / "data" / "anisotropy_spectrum.progress.log"

TEST = "--test" in sys.argv
N_STEPS = 512 if TEST else 4096
LAGS = [4, 8, 16, 32] if TEST else [8, 16, 32, 64, 128, 256]
SLOPES = np.linspace(-1.0, 1.0, 17 if TEST else 33)


def log(msg: str) -> None:
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with open(PROG_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def causal_mask(n_steps: int, n_cells: int, center: int) -> np.ndarray:
    rows = np.arange(n_steps, dtype=np.int32)[:, None]
    cols = np.arange(n_cells, dtype=np.int32)[None, :]
    return np.abs(cols - center) <= rows


def centered_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = a.astype(np.float64)
    y = b.astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denom = np.sqrt(np.mean(x * x) * np.mean(y * y))
    if denom == 0.0:
        return 0.0
    return float(np.mean(x * y) / denom)


def offset_corr(arr: np.ndarray, mask: np.ndarray, dt: int, dx: int) -> tuple[float, int]:
    if dx >= 0:
        a = arr[:-dt, : arr.shape[1] - dx]
        b = arr[dt:, dx:]
        valid = mask[:-dt, : arr.shape[1] - dx] & mask[dt:, dx:]
    else:
        shift = -dx
        a = arr[:-dt, shift:]
        b = arr[dt:, : arr.shape[1] - shift]
        valid = mask[:-dt, shift:] & mask[dt:, : arr.shape[1] - shift]
    if not np.any(valid):
        return 0.0, 0
    return centered_corr(a[valid], b[valid]), int(np.sum(valid))


def main() -> None:
    PROG_LOG.parent.mkdir(parents=True, exist_ok=True)
    open(PROG_LOG, "w").close()
    verify_spacetime_against_naive()
    log("Packed spacetime simulation verified against naive Rule 30.")

    center = N_STEPS
    initial = make_single_spike_row(2 * N_STEPS + 1, center)
    t0 = time.perf_counter()
    log(f"Simulating spacetime for anisotropy spectrum with N_STEPS={N_STEPS:,}, GPU={GPU_AVAILABLE}")
    bits = simulate_spacetime(initial, N_STEPS, gpu=GPU_AVAILABLE)
    mask = causal_mask(bits.shape[0], bits.shape[1], center)
    active_density = float(bits[mask].mean())
    rng = np.random.default_rng(20260401)
    random_bits = np.zeros_like(bits)
    random_bits[mask] = rng.binomial(1, active_density, size=int(mask.sum())).astype(np.uint8)

    corr_rule = np.zeros((len(LAGS), len(SLOPES)), dtype=np.float64)
    corr_rand = np.zeros_like(corr_rule)
    sample_counts = np.zeros_like(corr_rule, dtype=np.int32)
    offset_grid = np.zeros_like(corr_rule, dtype=np.int32)

    for i, dt in enumerate(LAGS):
        for j, slope in enumerate(SLOPES):
            dx = int(round(float(slope) * dt))
            offset_grid[i, j] = dx
            corr_rule[i, j], sample_counts[i, j] = offset_corr(bits, mask, dt, dx)
            corr_rand[i, j], _ = offset_corr(random_bits, mask, dt, dx)
        log(f"  processed lag dt={dt}")

    excess = corr_rule - corr_rand
    slope_summary = []
    for j, slope in enumerate(SLOPES):
        slope_summary.append(
            {
                "slope": round(float(slope), 6),
                "max_abs_excess_corr": round(float(np.max(np.abs(excess[:, j]))), 8),
                "mean_abs_excess_corr": round(float(np.mean(np.abs(excess[:, j]))), 8),
            }
        )

    elapsed = time.perf_counter() - t0
    result = {
        "date": datetime.date.today().isoformat(),
        "n_steps": N_STEPS,
        "n_cells": bits.shape[1],
        "gpu": GPU_AVAILABLE,
        "active_density": round(active_density, 8),
        "lags": LAGS,
        "slopes": [round(float(s), 6) for s in SLOPES],
        "offset_grid": offset_grid.tolist(),
        "corr_rule30": np.round(corr_rule, 8).tolist(),
        "corr_random": np.round(corr_rand, 8).tolist(),
        "corr_excess": np.round(excess, 8).tolist(),
        "sample_counts": sample_counts.tolist(),
        "slope_summary": slope_summary,
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

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            "Rule 30 anisotropy spectrum\n"
            f"{N_STEPS:,} steps, matched Bernoulli baseline",
            fontsize=12,
            fontweight="bold",
        )

        ax = axes[0]
        im = ax.imshow(
            corr_rule,
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-0.55,
            vmax=0.55,
        )
        ax.set_xticks(range(0, len(SLOPES), max(1, len(SLOPES) // 8)))
        ax.set_xticklabels([f"{SLOPES[k]:.2f}" for k in range(0, len(SLOPES), max(1, len(SLOPES) // 8))], rotation=45)
        ax.set_yticks(range(len(LAGS)))
        ax.set_yticklabels([str(x) for x in LAGS])
        ax.set_xlabel("Slope dx/dt")
        ax.set_ylabel("Lag dt")
        ax.set_title("Rule 30 directional correlation")
        plt.colorbar(im, ax=ax, label="Correlation")

        ax = axes[1]
        im = ax.imshow(
            excess,
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-0.55,
            vmax=0.55,
        )
        ax.set_xticks(range(0, len(SLOPES), max(1, len(SLOPES) // 8)))
        ax.set_xticklabels([f"{SLOPES[k]:.2f}" for k in range(0, len(SLOPES), max(1, len(SLOPES) // 8))], rotation=45)
        ax.set_yticks(range(len(LAGS)))
        ax.set_yticklabels([str(x) for x in LAGS])
        ax.set_xlabel("Slope dx/dt")
        ax.set_ylabel("Lag dt")
        ax.set_title("Excess over matched random baseline")
        plt.colorbar(im, ax=ax, label="Excess corr")

        ax = axes[2]
        ax.plot(SLOPES, [row["max_abs_excess_corr"] for row in slope_summary], "o-", color="darkgreen")
        ax.set_xlabel("Slope dx/dt")
        ax.set_ylabel("Max |excess corr| over lags")
        ax.set_title("Directional summary")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
        plt.close()
        log(f"Plot -> {PLOT_FILE}")
    except Exception as exc:
        log(f"Plot skipped: {exc}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Analyse the structure of right-side non-arrivals in causal sensitivity."""

from __future__ import annotations

import json
import math
import datetime
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\APATPROJECTS\rule30-research")
IN_JSON = ROOT / "data" / "causal_sensitivity.json"
OUT_JSON = ROOT / "data" / "right_blindspots.json"
PLOT_FILE = ROOT / "docs" / "plots" / "right_blindspots.png"


def contiguous_runs(values: list[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    runs: list[tuple[int, int]] = []
    start = values[0]
    prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = value
        prev = value
    runs.append((start, prev))
    return runs


def nearest_power_distance(values: np.ndarray) -> np.ndarray:
    max_value = int(values.max())
    powers = np.array([1 << k for k in range(int(math.ceil(math.log2(max_value + 1))) + 1)], dtype=np.int32)
    diffs = np.abs(values[:, None] - powers[None, :])
    return diffs.min(axis=1)


def prefix_occupancy(blind_set: set[int], max_dist: int, width: int) -> list[list[float]]:
    rows: list[list[float]] = []
    bit_width = max(1, max_dist.bit_length())
    for prefix_len in range(1, width + 1):
        bins = 1 << prefix_len
        counts = np.zeros(bins, dtype=np.int32)
        totals = np.zeros(bins, dtype=np.int32)
        for dist in range(1, max_dist + 1):
            code = dist >> max(0, bit_width - prefix_len)
            if code >= bins:
                code = bins - 1
            totals[code] += 1
            if dist in blind_set:
                counts[code] += 1
        with np.errstate(divide="ignore", invalid="ignore"):
            occ = np.divide(counts, totals, out=np.zeros_like(counts, dtype=np.float64), where=totals > 0)
        rows.append(occ.tolist())
    return rows


def main() -> None:
    with open(IN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    sim_length = int(data["sim_length"])
    max_dist = int(data["max_dist"])
    first_div_right = np.asarray(data["first_div_right"], dtype=np.int32)
    blind = np.flatnonzero(first_div_right == sim_length).astype(np.int32)
    blind_list = blind.tolist()
    blind_set = set(blind_list)
    blind_nozero = blind[blind > 0]
    gaps = np.diff(blind_nozero)
    runs = contiguous_runs(blind_nozero.tolist())
    run_lengths = [end - start + 1 for start, end in runs]

    residues: dict[str, dict[str, object]] = {}
    for modulus in [2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32]:
        counts = np.bincount(blind_nozero % modulus, minlength=modulus)
        expected = len(blind_nozero) / modulus
        max_dev = float(np.max(np.abs(counts - expected)))
        chi2 = float(np.sum((counts - expected) ** 2 / expected))
        residues[str(modulus)] = {
            "counts": counts.tolist(),
            "expected_each": expected,
            "max_abs_deviation": round(max_dev, 3),
            "chi2_uniform": round(chi2, 6),
        }

    near_power = nearest_power_distance(blind_nozero)
    rng = np.random.default_rng(20260401)
    baseline_counts = {str(radius): [] for radius in [0, 1, 2, 4, 8, 16]}
    all_distances = np.arange(1, max_dist + 1, dtype=np.int32)
    for _ in range(1000):
        sample = rng.choice(all_distances, size=len(blind_nozero), replace=False)
        sample_dist = nearest_power_distance(np.sort(sample))
        for radius in [0, 1, 2, 4, 8, 16]:
            baseline_counts[str(radius)].append(int(np.sum(sample_dist <= radius)))

    power_two_summary: dict[str, dict[str, float]] = {}
    for radius in [0, 1, 2, 4, 8, 16]:
        observed = int(np.sum(near_power <= radius))
        base = np.asarray(baseline_counts[str(radius)], dtype=np.float64)
        z = 0.0 if base.std() == 0 else float((observed - base.mean()) / base.std())
        power_two_summary[str(radius)] = {
            "observed": observed,
            "baseline_mean": round(float(base.mean()), 3),
            "baseline_std": round(float(base.std()), 3),
            "z_score": round(z, 3),
        }

    prefix_rows = prefix_occupancy(blind_set, max_dist, width=10)
    early_hits = np.flatnonzero((first_div_right <= max_dist) & (np.arange(max_dist + 1) > 0)).astype(np.int32)

    result = {
        "date": datetime.date.today().isoformat(),
        "input_file": str(IN_JSON),
        "sim_length": sim_length,
        "max_dist": max_dist,
        "blind_count": int(len(blind)),
        "blind_fraction": round(float(len(blind)) / (max_dist + 1), 8),
        "blind_examples_head": blind_list[:20],
        "blind_examples_tail": blind_list[-20:],
        "first_nontrivial_run": list(runs[0]) if runs else None,
        "longest_runs": [
            {"start": start, "end": end, "length": end - start + 1}
            for start, end in sorted(runs, key=lambda item: item[1] - item[0], reverse=True)[:10]
        ],
        "gap_histogram_top20": Counter(gaps.tolist()).most_common(20),
        "run_length_histogram_top20": Counter(run_lengths).most_common(20),
        "residue_tests": residues,
        "near_power_of_two": power_two_summary,
        "binary_prefix_occupancy": prefix_rows,
        "right_arrivals_examples": early_hits[:20].tolist(),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            "Right-side non-arrivals from causal sensitivity\n"
            f"{len(blind)} censored distances out of {max_dist + 1}",
            fontsize=12,
            fontweight="bold",
        )

        ax = axes[0, 0]
        indicator = np.zeros(max_dist + 1, dtype=np.uint8)
        indicator[blind] = 1
        ax.plot(np.arange(max_dist + 1), indicator, lw=0.8, color="black")
        for power in [1 << k for k in range(14)]:
            if power <= max_dist:
                ax.axvline(power, color="tomato", ls=":", lw=0.8, alpha=0.5)
        ax.set_xlim(0, max_dist)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Right-side distance")
        ax.set_ylabel("Censored by 10,000 steps")
        ax.set_title("Blind-spot indicator with powers of two")
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        gap_counts = Counter(gaps.tolist())
        top_gaps = sorted(gap_counts)[:20]
        ax.bar(top_gaps, [gap_counts[g] for g in top_gaps], color="steelblue")
        ax.set_xlabel("Gap to next blind distance")
        ax.set_ylabel("Count")
        ax.set_title("Gap distribution")
        ax.grid(True, alpha=0.3, axis="y")

        ax = axes[1, 0]
        image = np.full((len(prefix_rows), max(len(row) for row in prefix_rows)), np.nan, dtype=np.float64)
        for idx, row in enumerate(prefix_rows):
            image[idx, : len(row)] = row
        im = ax.imshow(image, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xlabel("Binary prefix bin")
        ax.set_ylabel("Prefix length")
        ax.set_yticks(range(len(prefix_rows)))
        ax.set_yticklabels([str(i) for i in range(1, len(prefix_rows) + 1)])
        ax.set_title("Blind-density by binary prefix")
        plt.colorbar(im, ax=ax, label="Blind fraction")

        ax = axes[1, 1]
        radii = [0, 1, 2, 4, 8, 16]
        observed = [power_two_summary[str(r)]["observed"] for r in radii]
        baseline = [power_two_summary[str(r)]["baseline_mean"] for r in radii]
        ax.plot(radii, observed, "o-", label="Observed", color="darkgreen")
        ax.plot(radii, baseline, "s--", label="Random sample baseline", color="gray")
        ax.set_xlabel("Distance to nearest power of two")
        ax.set_ylabel("Count within radius")
        ax.set_title("Clustering near powers of two")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"Plot skipped: {exc}")


if __name__ == "__main__":
    main()

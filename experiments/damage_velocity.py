#!/usr/bin/env python
"""Experiment Q — Directional Damage-Spreading Velocity.

A cleaner, interventional successor to Experiment M (causal_sensitivity) and the
right-blindspot follow-ups. Instead of reporting a censoring-prone "mean
asymmetry in steps", we measure the two *damage-spreading velocities* of Rule 30
directly.

Setup (per random initial condition):
  - Take a random initial row, evolve it -> reference center column.
  - Flip one cell at distance d to the LEFT or RIGHT of the center, evolve the
    perturbed row, and record the first step at which the center column diverges
    (the "damage arrival time"  first_div(d)).
  - Causality guarantees first_div(d) >= d (signal travels at most 1 cell/step).

Two directions:
  - LEFT-side flip  -> damage must travel RIGHTWARD to the center.
  - RIGHT-side flip -> damage must travel LEFTWARD  to the center.

The slope of median first_div(d) vs d gives 1/v for that direction. Averaging
over many random ICs removes the single-spike left-edge regularity that makes
Experiment M's left side look like a trivial T=D line, and tests whether the
asymmetry is a property of the *rule* rather than the spike IC.

Headline result (random ICs): rightward damage saturates the light cone
(v_right = 1.000, zero excess delay at every distance), while leftward damage
spreads at v_left ~ 0.245.  The asymmetry is intrinsic to Rule 30.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rule30_open_utils import simulate_center_columns_batch, GPU_AVAILABLE  # noqa: E402

TEST = "--test" in sys.argv

N_STEPS = 1200 if TEST else 4000
N_IC = 60 if TEST else 400
D_MAX = 120 if TEST else 300
D_STEP = 3
DS = list(range(1, D_MAX + 1, D_STEP))
SEEDS = [1] if TEST else [1, 2]

OUT_JSON = Path(__file__).resolve().parents[1] / "data" / "damage_velocity.json"


def measure(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    center = N_STEPS + D_MAX + 50
    n_cells = 2 * center + 1

    first_div = {"L": {d: [] for d in DS}, "R": {d: [] for d in DS}}
    for _ in range(N_IC):
        base = rng.integers(0, 2, size=n_cells, dtype=np.uint8)
        rows = [base]
        meta = []
        for side in ("L", "R"):
            for d in DS:
                r = base.copy()
                r[center + (d if side == "R" else -d)] ^= 1
                rows.append(r)
                meta.append((side, d))
        cols = simulate_center_columns_batch(
            np.asarray(rows), N_STEPS, center, gpu=GPU_AVAILABLE
        )
        ref = cols[0]
        for (side, d), c in zip(meta, cols[1:]):
            diff = np.flatnonzero(c != ref)
            first_div[side][d].append(int(diff[0]) if diff.size else N_STEPS + 1)

    out = {"seed": seed, "by_side": {}}
    for side in ("L", "R"):
        med = np.array([np.median(first_div[side][d]) for d in DS], dtype=float)
        dsa = np.array(DS, dtype=float)
        # Robust large-d fit (drop the first third where local-rule transients dominate)
        mask = dsa >= DS[len(DS) // 3]
        A = np.vstack([dsa[mask], np.ones(mask.sum())]).T
        slope, intercept = np.linalg.lstsq(A, med[mask], rcond=None)[0]
        excess = med - dsa  # first_div - d  (0 == saturates light cone)
        out["by_side"][side] = {
            "slope": round(float(slope), 4),
            "velocity": round(float(1.0 / slope), 4),
            "max_excess_delay": round(float(np.max(excess)), 2),
            "median_excess_at_dmax": round(float(excess[-1]), 2),
        }
    return out


def main():
    t0 = time.perf_counter()
    print(f"Experiment Q — Directional Damage-Spreading Velocity (TEST={TEST})")
    print(f"  GPU={GPU_AVAILABLE}  N_STEPS={N_STEPS}  N_IC={N_IC}  D_MAX={D_MAX}  seeds={SEEDS}")
    results = [measure(s) for s in SEEDS]
    for r in results:
        L, R = r["by_side"]["L"], r["by_side"]["R"]
        print(f"  seed {r['seed']}: "
              f"v_right(L-flip)={1/L['slope'] if L['slope'] else float('nan'):.3f} "
              f"(max excess {L['max_excess_delay']:.0f})  |  "
              f"v_left(R-flip)={R['velocity']:.3f} (slope {R['slope']:.3f})")

    elapsed = time.perf_counter() - t0
    payload = {
        "experiment": "Q_damage_velocity",
        "description": "Directional damage-spreading velocity of Rule 30 over random ICs.",
        "n_steps": N_STEPS, "n_ic": N_IC, "d_max": D_MAX, "d_step": D_STEP,
        "results": results,
        "elapsed_s": round(elapsed, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\nJSON -> {OUT_JSON}\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

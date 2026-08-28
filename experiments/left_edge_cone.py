#!/usr/bin/env python
"""Experiment R — Leftward Dependence-Cone Edge (interventional).

Supersedes the median-based velocity in Experiment Q, which conflated a *bulk*
arrival rate (median first-divergence) with a *causal* light-cone edge. For the
prize problems the causal edge is the quantity that matters: it bounds which
initial cells the center column can possibly depend on.

Theory anchor (so the two directions are not treated symmetrically):
  Rule 30 ANF:  new[i] = a[i-1] XOR a[i] XOR a[i+1] XOR (a[i] AND a[i+1]).
  The left neighbour a[i-1] enters as a PURE XOR  =>  the rule is LEFT-PERMUTIVE
  (a bijection in its leftmost input). Consequences we rely on:
    * RIGHTWARD damage travels at exactly speed 1 on every background -> v_R = 1
      is a THEOREM, not a measurement (we only sanity-check it here).
    * The uniform Bernoulli(1/2) measure is invariant, so a random i.i.d. initial
      row stays i.i.d. fair at every later time => the center column is PROVABLY
      50/50 for random ICs. Prize-3 equidistribution is therefore only open for
      the single deterministic seed ...0001000...
  The right neighbour sits inside the OR/AND term, so LEFTWARD propagation is
  conditional (gated on a[i]=0). v_L is the only empirically interesting speed.

What this experiment measures (interventional, with a null):
  For a sweep of distances d, over many random ICs, flip the initial cell at
  distance d to the RIGHT of the center and record first_div(d) = first step the
  center column changes. Causality => first_div(d) >= d (speed-1 null edge).
  We report the EDGE (fastest arrival = min / low percentiles) and the BULK
  (median), and test whether the edge slope converges to a hard 1/lambda_L < 1
  as d grows. A convergent edge slope > 1 is a real frozen-cone result:
  the center cannot depend on right-side seed cells beyond ~lambda_L * T.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rule30_open_utils import (  # noqa: E402
    simulate_center_columns_batch,
    simulate_naive_center_columns,
    GPU_AVAILABLE,
)

TEST = "--test" in sys.argv

DS = [30, 60, 120, 240] if TEST else [30, 60, 120, 240, 480, 960]
N_IC = 200 if TEST else 1500
# horizon must comfortably exceed d / edge_speed; edge ~0.4 so d/0.4 = 2.5d, use 6x
HORIZON_MULT = 6
SEED = 11

OUT_JSON = Path(__file__).resolve().parents[1] / "data" / "left_edge_cone.json"


def naive_crosscheck(rng) -> None:
    """Confirm the packed GPU/CPU kernel matches the naive rule for right-flips."""
    n_steps, center = 220, 320
    n_cells = 2 * center + 1
    base = rng.integers(0, 2, size=n_cells, dtype=np.uint8)
    for d in (1, 5, 17, 40):
        r = base.copy()
        r[center + d] ^= 1
        fast = simulate_center_columns_batch(np.stack([base, r]), n_steps, center, gpu=GPU_AVAILABLE)
        naive = simulate_naive_center_columns(np.stack([base, r]), n_steps, center)
        if not np.array_equal(fast, naive):
            raise RuntimeError(f"packed != naive at d={d}")


def arrivals_for_d(d: int, rng) -> np.ndarray:
    n_steps = int(d * HORIZON_MULT)
    center = n_steps + d + 50
    n_cells = 2 * center + 1
    out = np.empty(N_IC, dtype=np.int64)
    # batch ICs to keep VRAM modest: 2 rows (ref+flip) per IC, do in chunks
    chunk = max(1, min(N_IC, 4_000_000 // max(1, n_cells)))
    filled = 0
    while filled < N_IC:
        k = min(chunk, N_IC - filled)
        rows = np.empty((2 * k, n_cells), dtype=np.uint8)
        bases = rng.integers(0, 2, size=(k, n_cells), dtype=np.uint8)
        for j in range(k):
            rows[2 * j] = bases[j]
            r = bases[j].copy()
            r[center + d] ^= 1
            rows[2 * j + 1] = r
        cols = simulate_center_columns_batch(rows, n_steps, center, gpu=GPU_AVAILABLE)
        for j in range(k):
            diff = np.flatnonzero(cols[2 * j + 1] != cols[2 * j])
            out[filled + j] = int(diff[0]) if diff.size else n_steps + 1
        filled += k
    return out


def main():
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    print(f"Experiment R - Leftward Dependence-Cone Edge (TEST={TEST}, GPU={GPU_AVAILABLE})")
    naive_crosscheck(rng)
    print("  naive cross-check OK")

    rows_out = []
    print(f"  {'d':>5} {'cens':>6} {'min':>7} {'p1':>7} {'p10':>7} {'med':>7} "
          f"{'v_edge':>7} {'v_p10':>7} {'v_med':>7}")
    for d in DS:
        arr = arrivals_for_d(d, rng)
        n_steps = int(d * HORIZON_MULT)
        cens = float(np.mean(arr > n_steps))
        mn = int(arr.min())
        p1 = float(np.percentile(arr, 1))
        p10 = float(np.percentile(arr, 10))
        med = float(np.median(arr))
        rec = {
            "d": d, "n_steps": n_steps, "censored_frac": round(cens, 4),
            "min": mn, "p1": p1, "p10": p10, "median": med,
            "v_edge": round(d / mn, 4),
            "v_p10": round(d / p10, 4),
            "v_median": round(d / med, 4),
        }
        rows_out.append(rec)
        print(f"  {d:5d} {cens:6.3f} {mn:7d} {p1:7.0f} {p10:7.0f} {med:7.0f} "
              f"{rec['v_edge']:7.3f} {rec['v_p10']:7.3f} {rec['v_median']:7.3f}")

    # Convergence read-out: do the edge / p10 / median speeds settle to a constant?
    big = [r for r in rows_out if r["d"] >= DS[len(DS) // 2]]
    conv = {
        "v_edge_largest_d": rows_out[-1]["v_edge"],
        "v_p10_largest_d": rows_out[-1]["v_p10"],
        "v_median_largest_d": rows_out[-1]["v_median"],
        "v_median_mean_over_large_d": round(float(np.mean([r["v_median"] for r in big])), 4),
        "v_p10_mean_over_large_d": round(float(np.mean([r["v_p10"] for r in big])), 4),
    }
    elapsed = time.perf_counter() - t0
    payload = {
        "experiment": "R_left_edge_cone",
        "note": "Interventional leftward dependence-cone edge. v_right=1 is a "
                "left-permutivity theorem (checked, not measured). Edge = fastest "
                "arrival (causal bound); median = bulk damage rate.",
        "n_ic": N_IC, "horizon_mult": HORIZON_MULT, "seed": SEED,
        "per_d": rows_out, "convergence": conv, "elapsed_s": round(elapsed, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print("\nConvergence:")
    for k, v in conv.items():
        print(f"  {k} = {v}")
    print(f"\nJSON -> {OUT_JSON}\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

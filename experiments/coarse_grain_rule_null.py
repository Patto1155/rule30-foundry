#!/usr/bin/env python
"""Experiment T2 — Same-Local-Statistics Null for the Coarse-Grain Search.

Question
--------
Experiment T found Rule 30's best b=2 coarse field "closes" with excess ~0.22
over an i.i.d. fair-coin null. But an i.i.d. coin is a *weak* null: ANY
locally-correlated field beats it, because neighbouring coarse cells share fine
cells and inherit local rule structure. So the ~0.22 excess might be generic
"local leakage" present in every chaotic elementary CA — not evidence that Rule
30 is specially (approximately) reducible.

Design
------
Run the IDENTICAL coarse-grain pipeline (sheared b=2 enumeration, excess closure
over the optimal local predictor) on several rules and compare:

  * rule 30  — the subject (chaotic, class 3, the canonical irreducible rule)
  * rule 45  — SAME-STATISTICS NULL: also chaotic/class-3, ~balanced density;
               if 30 ≈ 45 the excess is generic local leakage, not 30-specific.
  * rule 90  — POSITIVE CONTROL: additive/linear (XOR of neighbours), genuinely
               reducible. A meaningful closure metric should rate 90 ABOVE 30;
               if it does not, the metric itself is too weak to detect
               reducibility and the whole probe is uninformative.
  * rule 110 — class-4, Turing-complete: a different complexity regime.
  * i.i.d.   — the original weak null, for reference.

Read-out
--------
  - 30 ≈ 45  and  90 not clearly higher  -> excess is generic local leakage;
    Rule 30 is NOT specially reducible at b=2 (consistent with Israeli-Goldenfeld
    irreducibility). The b=2 route is closed; the open question moves to b>=3.
  - 90 >> {30,45}  -> the metric DOES separate reducible from irreducible, and we
    can trust its (negative) verdict on Rule 30.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coarse_grain_search import enumerate_b2  # noqa: E402  (reuse exact pipeline)
from eca_sim import simulate_spacetime_rule, GPU_AVAILABLE  # noqa: E402

OUT_JSON = Path(__file__).resolve().parents[1] / "data" / "coarse_grain_rule_null.json"

RULES = {
    30: "subject (chaotic, class 3)",
    45: "same-statistics null (chaotic, class 3)",
    90: "positive control (additive/linear -> reducible)",
    110: "class-4, Turing-complete",
}


def bulk_rule_field(rule: int, n_steps: int, width: int, seed: int) -> np.ndarray:
    """Random-IC interior window of a rule's spacetime (boundary trimmed)."""
    rng = np.random.default_rng(seed)
    margin = n_steps + 32
    n_cells = width + 2 * margin
    row = rng.integers(0, 2, size=n_cells, dtype=np.uint8)
    st = simulate_spacetime_rule(row, n_steps, rule, gpu=GPU_AVAILABLE)
    return st[:, margin:margin + width].copy()


def best_over_shears(field: np.ndarray, shears, r: int, h_min: float):
    """Best (max-excess) coarse closure over the given shears for one field."""
    best = {"excess": -1.0, "closure": 0.0, "shear": None}
    per_shear = {}
    for sigma in shears:
        g = enumerate_b2(field, sigma, r, h_min)
        per_shear[sigma] = {"excess": round(g["excess"], 4), "closure": round(g["closure"], 4)}
        if g["excess"] > best["excess"]:
            best = {"excess": g["excess"], "closure": g["closure"], "shear": sigma}
    return best, per_shear


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    n_steps = 400 if args.test else 1200
    width = 400 if args.test else 1200
    r = 1
    h_min = 0.85
    shears = [0.0, 0.25, 1.0]      # axis-aligned + two cone-aligned
    seed = 7

    print(f"Experiment T2 - Same-statistics null (GPU={GPU_AVAILABLE}, test={args.test})")
    print(f"  field {n_steps}x{width}, b=2, r={r}, H_min={h_min}, shears={shears}")
    print(f"  EXCESS = best coarse closure - marginal baseline (i.i.d. ~ 0)\n")
    print(f"  {'rule':>5} | {'excess':>7} {'closure':>7} {'shear':>5} | role")

    t0 = time.perf_counter()
    results = {}

    # Real rules.
    for rule, role in RULES.items():
        field = bulk_rule_field(rule, n_steps, width, seed)
        best, per_shear = best_over_shears(field, shears, r, h_min)
        results[str(rule)] = {"role": role, "density": round(float(field.mean()), 4),
                              "best": best, "per_shear": per_shear}
        print(f"  {rule:>5} | {best['excess']:>+7.4f} {best['closure']:>7.4f} "
              f"{str(best['shear']):>5} | {role}")

    # i.i.d. null (same shape).
    null = np.random.default_rng(99).integers(0, 2, size=(n_steps, width), dtype=np.uint8)
    nb, nps = best_over_shears(null, shears, r, h_min)
    results["iid"] = {"role": "weak null (fair coin)", "best": nb, "per_shear": nps}
    print(f"  {'iid':>5} | {nb['excess']:>+7.4f} {nb['closure']:>7.4f} "
          f"{str(nb['shear']):>5} | weak null (fair coin)")

    e30 = results["30"]["best"]["excess"]
    e45 = results["45"]["best"]["excess"]
    e90 = results["90"]["best"]["excess"]
    eiid = results["iid"]["best"]["excess"]

    gap_30_45 = e30 - e45
    gap_90_30 = e90 - e30

    if e90 <= e30 + 0.02:
        verdict = ("Metric does NOT separate reducible (rule 90) from irreducible "
                   "(rule 30): the b=2 excess-closure probe is too weak to be "
                   "informative about reducibility. Need a stronger probe.")
    elif abs(gap_30_45) <= 0.03:
        verdict = ("Rule 30 ~= rule 45 (chaotic null) while rule 90 (linear) stands "
                   "out: the ~0.2 excess is GENERIC local leakage, not 30-specific "
                   "reducibility. Rule 30 is irreducible at b=2; open question -> b>=3.")
    else:
        verdict = (f"Rule 30 excess separates from the chaotic null by {gap_30_45:+.3f} "
                   "AND the metric is validated by rule 90 standing out: possible "
                   "30-specific structure at b=2 - investigate.")

    payload = {
        "experiment": "T2_same_statistics_null",
        "params": {"n_steps": n_steps, "width": width, "b": 2, "r": r,
                   "h_min": h_min, "shears": shears, "seed": seed, "test": args.test},
        "results": results,
        "summary": {"excess_30": round(e30, 4), "excess_45": round(e45, 4),
                    "excess_90": round(e90, 4), "excess_iid": round(eiid, 4),
                    "gap_30_minus_45": round(gap_30_45, 4),
                    "gap_90_minus_30": round(gap_90_30, 4)},
        "verdict": verdict,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\n  30-45 gap = {gap_30_45:+.4f}   90-30 gap = {gap_90_30:+.4f}")
    print(f"  Verdict: {verdict}")
    print(f"\nJSON -> {OUT_JSON}\nDone in {payload['elapsed_s']}s")


if __name__ == "__main__":
    main()

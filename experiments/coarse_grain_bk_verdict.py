#!/usr/bin/env python
"""Experiment U — b=3 coarse-grain reducibility verdict (searched, with gate+null).

Extends the b=2 verdict (Experiment T2) to b=3, where enumeration is impossible
(2^512 projections) so closure is maximized by GPU-resident search
(coarse_grain_bk.search_projection).

Discipline (a SEARCH, not an enumeration, so two extra controls are mandatory):

  * Validity gate -- a SHIFT rule (170): its coarse field is exactly
    coarse[t+1,x]=coarse[t,x+1] for ANY projection, so closure must reach ~1.0 at
    any b/r. If the search can't recover it, the search is too weak to trust its
    negative on Rule 30. (NOTE: linear rules 90/150 are NOT valid b=3 controls --
    they are not exactly coarse-grainable at r=1.)
  * Searched i.i.d. null -- a search overfits finite samples, so the i.i.d. field
    must be SEARCHED at the same budget/M; its closure is the overfitting floor
    that Rule 30 must clear to mean anything.
  * Same-statistics null -- rule 45 (chaotic, class 3) searched at equal budget.
    If rule 30 ~= rule 45, any excess is generic chaotic structure, not
    30-specific reducibility.

Best projection is re-scored on the FULL (uncapped) transition set for the
reported closure. Read-out: 30 ~= 45 (both near the floor) -> b=3 route closed;
30 clearly above 45 with the gate passed -> candidate reducibility, investigate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eca_sim import simulate_spacetime_rule, GPU_AVAILABLE  # noqa: E402
from coarse_grain_bk import prep_blocks, search_projection, score_projection  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_JSON = DATA_DIR / "coarse_grain_b3_verdict.json"
TEST_OUT_JSON = DATA_DIR / "coarse_grain_b3_verdict_test.json"


def _field(rule, n_steps, width, seed):
    rng = np.random.default_rng(seed)
    margin = n_steps + 32
    nc = width + 2 * margin
    row = rng.integers(0, 2, size=nc, dtype=np.uint8)
    if rule == "iid":
        return rng.integers(0, 2, size=(n_steps, width), dtype=np.uint8)
    st = simulate_spacetime_rule(row, n_steps, rule, gpu=GPU_AVAILABLE)
    return st[:, margin:margin + width].copy()


def _round_optional(value, ndigits=5):
    return None if value is None else round(value, ndigits)


def search_best_over_shears(field, b, shears, r, budget, msearch, hmin, sseed):
    best = {"full_closure": -1.0, "full_excess": -1.0, "shear": None}
    per_shear = {}
    for sigma in shears:
        prep = prep_blocks(field, b, sigma, r, gpu=GPU_AVAILABLE, m_max=msearch, seed=0)
        res = search_projection(prep, budget=budget, b=b, seed=sseed, h_min=hmin)
        prep_full = prep_blocks(field, b, sigma, r, gpu=GPU_AVAILABLE)
        full = score_projection(prep_full, res["best_pi"], h_min=hmin)
        fc = full["closure"]
        fe = full["excess"] if full["valid"] else None
        per_shear[str(sigma)] = {"full_closure": round(fc, 5), "full_excess": _round_optional(fe),
                                 "full_valid": full["valid"],
                                 "full_entropy": round(full["entropy"], 5),
                                 "search_closure": round(res["best_closure"], 5),
                                 "evals": res["evals"]}
        gated_closure = fc if full["valid"] else -1.0
        if gated_closure > best["full_closure"]:
            best = {"full_closure": round(fc, 5), "full_excess": _round_optional(fe), "shear": sigma,
                    "full_valid": full["valid"], "full_entropy": round(full["entropy"], 5)}
    return best, per_shear


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--b", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--width", type=int, default=1500)
    ap.add_argument("--budget", type=int, default=300000)
    ap.add_argument("--msearch", type=int, default=80000)
    ap.add_argument("--shears", default="0,0.25,1")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sseed", type=int, default=1)
    ap.add_argument("--out", type=Path,
                    help="JSON output path. Defaults to the full verdict path; --test uses a test path.")
    args = ap.parse_args()
    if args.test:
        args.steps = args.width = 600
        args.budget = 15000
        args.msearch = 40000
    out_json = args.out or (TEST_OUT_JSON if args.test else OUT_JSON)

    shears = [float(x) for x in args.shears.split(",")]
    r = 1
    hmin = 0.85
    entities = [("control_shift_170", 170), ("iid", "iid"),
                ("rule_30", 30), ("rule_45", 45), ("rule_110", 110)]

    print(f"Experiment U - b={args.b} coarse-grain verdict (GPU={GPU_AVAILABLE}, test={args.test})")
    print(f"  steps={args.steps} width={args.width} budget={args.budget} "
          f"msearch={args.msearch} shears={shears}")
    print(f"  {'entity':>18} | {'closure':>8} {'excess':>8} {'shear':>5}")

    t0 = time.perf_counter()
    results = {}
    for name, rule in entities:
        field = _field(rule, args.steps, args.width, args.seed)
        best, per_shear = search_best_over_shears(field, args.b, shears, r,
                                                  args.budget, args.msearch, hmin, args.sseed)
        results[name] = {"rule": rule, "best": best, "per_shear": per_shear}
        print(f"  {name:>18} | {best['full_closure']:>8.4f} {best['full_excess']:>+8.4f} "
              f"{str(best['shear']):>5}")

    gate = results["control_shift_170"]["best"]["full_closure"]
    floor = results["iid"]["best"]["full_closure"]
    c30 = results["rule_30"]["best"]["full_closure"]
    c45 = results["rule_45"]["best"]["full_closure"]
    gate_ok = gate >= 0.99
    gap_30_45 = c30 - c45

    # Only a POSITIVE gap (rule 30 ABOVE the chaotic null) is evidence for
    # 30-specific reducibility. A gap <= +0.02 -- including negative, i.e. rule 30
    # at/below the null -- means no special reducibility (route closed).
    if not gate_ok:
        verdict = (f"GATE FAILED (shift control only {gate:.3f} < 0.99): the search is "
                   "too weak to trust any negative. Increase budget/operators.")
    elif gap_30_45 > 0.02:
        verdict = (f"Rule 30 ({c30:.3f}) closes ABOVE rule 45 ({c45:.3f}) by "
                   f"{gap_30_45:+.3f} with gate OK ({gate:.3f}): candidate b=3 "
                   "reducibility -- investigate.")
    else:
        rel = "~=" if abs(gap_30_45) <= 0.02 else "<="
        verdict = (f"Rule 30 ({c30:.3f}) {rel} rule 45 ({c45:.3f}) (gap {gap_30_45:+.3f}); "
                   f"gate OK ({gate:.3f}), i.i.d. floor {floor:.3f}. Rule 30 is NOT above the "
                   "chaotic null -- any excess is generic chaotic structure, not 30-specific. "
                   "b=3 coarse-grain route CLOSED.")

    payload = {"experiment": "U_b3_coarse_grain_verdict",
               "params": {"b": args.b, "steps": args.steps, "width": args.width,
                          "budget": args.budget, "msearch": args.msearch, "shears": shears,
                          "r": r, "hmin": hmin, "seed": args.seed, "sseed": args.sseed,
                          "gpu": GPU_AVAILABLE, "test": args.test},
               "results": results,
               "summary": {"gate_shift": round(gate, 4), "gate_ok": gate_ok,
                           "iid_floor": round(floor, 4), "closure_30": round(c30, 4),
                           "closure_45": round(c45, 4), "gap_30_minus_45": round(gap_30_45, 4)},
               "verdict": verdict, "elapsed_s": round(time.perf_counter() - t0, 1)}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\n  gate={gate:.4f} (ok={gate_ok})  floor={floor:.4f}  "
          f"30={c30:.4f}  45={c45:.4f}  gap={gap_30_45:+.4f}")
    print(f"  Verdict: {verdict}")
    print(f"\nJSON -> {out_json}\nDone in {payload['elapsed_s']}s")


if __name__ == "__main__":
    main()

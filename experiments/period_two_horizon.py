#!/usr/bin/env python
"""Does the sharp-horizon method reach eventual period two? (Prize 1, partial)

Condrey (arXiv:2609.09431v1, 8 Sep 2026, preprint) classifies the two constant
trace fibers of Rule 30 and derives a sharp horizon: writing H(p, w) for the
longest eventually p-periodic centre prefix over nonzero rows of support radius
w, Theorems 6 and 7 with Corollary 8 give

    H(1, w) = w + 2   (prefix length; the horizon INDEX is w + 1)

from which Corollary 5 follows -- no column of a nonzero finite Rule 30 orbit
is eventually constant, so Prize 1 is settled for p = 1. The obvious next rung
is p = 2, which the paper's section 5 explicitly leaves open and which Kari,
thanked in its acknowledgments, suggested generalising to all nonconstant
periodic traces.

This script decides one question before any effort is spent on that rung:
**does the horizon method itself extend?** Condrey section 5 says it does not,
and gives the reason in one sentence -- left permutivity supplies, for any
prescribed trace of length N and any compatible right prefix, a unique left
half, hence a finite row of support radius about N realising that trace through
time N, so H(2, w) >= w for every w, and "no bounded law can exist".

That is a claim in an unrefereed preprint about a method this repo would
otherwise build on, so it is checked here rather than believed -- and it does
not survive the check. **H(1, w) = w + 2 is also linear in w and also at least
w**, so the stated reason is satisfied by the very case the method settles. A
linear lower bound does not distinguish p = 2 from p = 1 and therefore closes
nothing. What would close the method is a finite row whose alternating trace
never breaks, at some fixed radius; that is what this script looks for.

Two parts:

1. **Reproduce the published p = 1 table exactly** (appendix, w = 1..7): the
   maximum horizon index and the extremizer count for each of x_0 = 0 and
   x_0 = 1. This is the positive control. An implementation that cannot
   reproduce a sharp published law has not earned the right to report a new
   one, and a mismatch here is a bug in this file, not a refutation of a
   theorem.

2. **Measure H(2, w)**, exhaustively over every nonzero row of support radius
   w, for the ALTERNATING trace (period exactly two) as well as for period
   dividing two. Constant traces are period-two as well and are already
   classified, so they are reported separately and never conflated.

If H(2, w) grows with w, the horizon route to period two is closed and the
`nonconstant-periodic-trace` obligation must be reached some other way. That
is a route-pricing result, not a prize result: excluding p = 2 alone would
still leave every p >= 3 open, and Prize 1 needs all of them.

    python experiments/period_two_horizon.py --pretty
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.rule30_open_utils import step_naive_open  # noqa: E402

# Condrey, appendix table: (w, H0, count0, H1, count1). Horizon INDEX, so the
# corresponding prefix length is the index plus one.
PUBLISHED_H1_TABLE = {
    1: (2, 1, 1, 2), 2: (2, 3, 3, 4), 3: (4, 7, 3, 8), 4: (4, 15, 5, 16),
    5: (6, 31, 5, 32), 6: (6, 63, 7, 64), 7: (8, 127, 7, 128),
}


def all_rows(w: int, steps: int):
    """Every nonzero row of support radius <= w, padded so the cone never clips.

    Returned as one (n_configs, width) array so the whole family advances in
    lockstep under numpy rather than one Python loop per configuration.
    """
    span = 2 * w + 1
    n = 1 << span
    width = 2 * (w + steps) + 3
    centre = width // 2
    idx = np.arange(n, dtype=np.uint32)
    bits = ((idx[:, None] >> np.arange(span, dtype=np.uint32)[None, :]) & 1).astype(np.uint8)
    keep = bits.any(axis=1)
    rows = np.zeros((int(keep.sum()), width), dtype=np.uint8)
    rows[:, centre - w: centre + w + 1] = bits[keep]
    return rows, centre


def step_rows(rows: np.ndarray) -> np.ndarray:
    left = np.zeros_like(rows)
    left[:, 1:] = rows[:, :-1]
    right = np.zeros_like(rows)
    right[:, :-1] = rows[:, 1:]
    return left ^ (rows | right)


def traces(w: int, steps: int) -> np.ndarray:
    rows, centre = all_rows(w, steps)
    out = np.empty((rows.shape[0], steps + 1), dtype=np.uint8)
    for t in range(steps + 1):
        out[:, t] = rows[:, centre]
        if t < steps:
            rows = step_rows(rows)
    return out


def self_test(w: int = 3, steps: int = 20) -> dict:
    """Vectorised family against the repo's naive single-row stepper."""
    rows, centre = all_rows(w, steps)
    tr = traces(w, steps)
    mismatch = 0
    for i in (0, 1, rows.shape[0] // 2, rows.shape[0] - 1):
        row = rows[i].copy()
        for t in range(steps + 1):
            if int(row[centre]) != int(tr[i, t]):
                mismatch += 1
            if t < steps:
                row = step_naive_open(row)
    return {"rows_checked": 4, "steps": steps, "mismatches": mismatch}


def constant_horizons(w: int, steps: int) -> dict:
    """H_0 and H_1: longest constant centre prefix, by initial centre value."""
    tr = traces(w, steps)
    first = tr[:, 0]
    out = {}
    for value, name in ((0, "H0"), (1, "H1")):
        sel = tr[first == value]
        if not len(sel):
            out[name] = None
            continue
        differs = sel != value
        # Horizon index = (first differing time) - 1, or steps if never differs.
        any_diff = differs.any(axis=1)
        first_diff = np.where(any_diff, differs.argmax(axis=1), steps + 1)
        horizon = first_diff - 1
        best = int(horizon.max())
        out[name] = {"max_index": best, "count": int((horizon == best).sum()),
                     "saturated": bool(best >= steps)}
    return out


def period_two_horizon(w: int, steps: int) -> dict:
    """Longest centre prefix satisfying c_t = c_{t+2}, and the alternating case.

    `period_dividing_two` includes the constant traces, which are already
    classified; `alternating` is the genuinely new, nonconstant case.
    """
    tr = traces(w, steps)
    n, T = tr.shape
    violate = tr[:, :-2] != tr[:, 2:]          # position j means c_j != c_{j+2}
    any_v = violate.any(axis=1)
    first_v = np.where(any_v, violate.argmax(axis=1), T)
    # A violation at j means the prefix is period-2 up to index j+1.
    horizon = np.minimum(first_v + 1, T - 1)

    alternating = tr[:, 0] != tr[:, 1]
    result = {}
    for mask, name in ((np.ones(n, bool), "period_dividing_two"),
                       (alternating, "alternating")):
        h = horizon[mask]
        if not len(h):
            result[name] = None
            continue
        best = int(h.max())
        result[name] = {"max_index": best, "prefix_length": best + 1,
                        "count": int((h == best).sum()),
                        "saturated": bool(best >= T - 1)}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-radius", type=int, default=8)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--out", type=Path)
    p.add_argument("--pretty", action="store_true")
    args = p.parse_args()

    check = self_test()
    if check["mismatches"]:
        print(f"period_two_horizon: vectorised family disagrees with the naive "
              f"stepper ({check['mismatches']} cells); refusing to report", file=sys.stderr)
        return 2
    if args.self_test:
        print(json.dumps(check, indent=2))
        return 0

    rows = []
    control_ok = True
    for w in range(1, args.max_radius + 1):
        const = constant_horizons(w, args.steps)
        p2 = period_two_horizon(w, args.steps)
        entry = {"radius": w, "constant": const, "period_two": p2}
        if w in PUBLISHED_H1_TABLE:
            h0, c0, h1, c1 = PUBLISHED_H1_TABLE[w]
            got = (const["H0"]["max_index"], const["H0"]["count"],
                   const["H1"]["max_index"], const["H1"]["count"])
            entry["published_p1_match"] = (got == (h0, c0, h1, c1))
            entry["published_p1_expected"] = [h0, c0, h1, c1]
            entry["published_p1_observed"] = list(got)
            control_ok &= entry["published_p1_match"]
        rows.append(entry)

    alt = [(r["radius"], r["period_two"]["alternating"]["prefix_length"],
            r["period_two"]["alternating"]["saturated"]) for r in rows]
    growing = all(b >= a for (_, a, _), (_, b, _) in zip(alt, alt[1:]))
    saturated = [w for w, _, sat in alt if sat]

    # The comparison that decides anything. Condrey section 5 argues the method
    # fails at p=2 because left permutivity forces H(2,w) >= w, so "no bounded
    # law can exist". That reason does not separate the two cases: the p=1 law
    # the same paper proves is H(1,w) = w + 2, which is also linear in w and is
    # also >= w. A linear lower bound is therefore compatible with exactly the
    # kind of finite sharp horizon the p=1 proof uses. What would close the
    # method is an alternating horizon that is unbounded at some FIXED radius --
    # a finite row with an infinite alternating trace -- not one that grows.
    p1_prefix = {r["radius"]: r["radius"] + 2 for r in rows}
    ratio = [[w, n, p1_prefix[w], round(n / p1_prefix[w], 3)] for w, n, _ in alt]
    report = {
        "steps": args.steps,
        "self_test": check,
        "positive_control_reproduces_published_p1_table": control_ok,
        "rows": rows,
        "alternating_prefix_length_by_radius": [[w, n] for w, n, _ in alt],
        "alternating_non_decreasing_in_radius": bool(growing),
        "alternating_vs_p1_prefix": ratio,
        "saturated_radii": saturated,
        "verdict": (
            "THE HORIZON ROUTE TO PERIOD TWO IS NOT CLOSED, and the growth of "
            "H(2, w) does not close it. The published p=1 law H(1, w) = w + 2 is "
            "itself linear in w, so 'H(2, w) >= w' -- the reason given for the "
            "method failing -- is satisfied by the case the method DOES settle "
            "and separates nothing. Every radius measured here has a FINITE "
            "alternating horizon, which is the same shape as the p=1 law at a "
            "different slope. What would actually close the method is a finite "
            "row with an infinite alternating trace; none was found."
            if not saturated else
            "Some radii saturated the step budget: their horizons are lower "
            "bounds, not maxima. Re-run with more steps before reading anything "
            "into the law."),
        "claim_limit": (
            "This prices a method; it proves nothing about the prize. Excluding "
            "eventual period 2 would still leave every p >= 3 open, and Prize 1 "
            "needs all periods. Reproducing the published p=1 table is a check on "
            "this implementation, not new evidence about period two. A horizon "
            "that saturates at the step budget is reported as saturated and is a "
            "lower bound, not a measured maximum."),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2) if args.pretty else json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

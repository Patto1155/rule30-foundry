"""Backward dependency trace for a single left-diagonal query.

The left diagonals satisfy a closed recursion (docs/theory/README.md §3):

    D_d(t+1) = D_{d-2}(t) XOR ( D_{d-1}(t) OR D_d(t) )                   (R)

`experiments/transient_branch_selector.py` established that at a settled zero
word the successor is selected by a few bits, but computing those bits still
appeared to need the whole transient. That is a statement about *replaying the
transient forwards*. This script asks the complementary question:

    evaluating ONE cell backwards through (R), with memoisation and with the
    OR short-circuited, how many cells must actually be evaluated?

The short-circuit is the point. In (R), `D_d(t)` enters only inside the OR, so
whenever `D_{d-1}(t) = 1` the OR is 1 whatever `D_d(t)` is, and that entire
sub-cone is never needed. This is an exact algebraic cancellation, not an
approximation: the value returned is bit-identical to the simulator's.

Three costs are reported, all exact counts of distinct cells:

  cells_rect       forward fill of the rectangle [0..d] x [0..t]  -- what a
                   rolling propagator pays
  cells_cone       the backward dependency closure with NO short-circuit
  cells_evaluated  the same closure WITH the short-circuit

`useful success` would be cells_evaluated / cells_cone shrinking with depth,
which would suggest a scaling conjecture. `useful failure` is the ratio sitting
at a constant, which says the cancellation buys a constant factor and no more.
All preprocessing is charged to the algorithm: there is no precomputed table.

Usage:
    python experiments/branch_dependency_trace.py --depths 64,128,256 --pretty
    python experiments/branch_dependency_trace.py --verify runs/<artifact>.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ARTIFACT_TYPE = "rule30.branch_dependency_trace"
ARTIFACT_VERSION = 1

# D_d(t) = a(t, -t+d). At t=0 the row is the lone seed cell a(0,0), so
# D_d(0) = [d == 0]. The cone edge gives D_0 == 1 and D_1(t>=1) == 1.
# Outside the cone (i = -t+d > t, i.e. d > 2t) every cell is 0.


def reference_diagonals(steps: int, diagonals: int) -> np.ndarray:
    """(steps+1, diagonals) array of D_d(t), by direct Rule 30 simulation.

    The independent oracle for every value this module computes. Deliberately
    simple: one big-int row, shifted, no packing conventions to get wrong.
    """
    width = 2 * steps + 3
    centre = steps + 1
    mask = (1 << width) - 1
    out = np.zeros((steps + 1, diagonals), dtype=np.uint8)
    state = 1 << centre
    for t in range(steps + 1):
        win = state >> (centre - t)
        for d in range(diagonals):
            out[t, d] = (win >> d) & 1
        state = ((state << 1) ^ (state | (state >> 1))) & mask
    return out


def _base(d: int, t: int) -> int | None:
    """Value of D_d(t) when it needs no recursion, else None."""
    if d < 0 or t < 0:
        raise ValueError(f"out of range: d={d}, t={t}")
    if d > 2 * t:
        return 0
    if d == 0:
        return 1
    if t == 0:
        return 1 if d == 0 else 0
    if d == 1:
        return 1
    return None


def trace(d: int, t: int, short_circuit: bool = True) -> dict:
    """Evaluate D_d(t) backwards through (R). Returns the value and the cost.

    Explicit stack rather than recursion: the closure is ~t deep and Python's
    limit is not the thing under test. `memo` doubles as the evaluated-cell
    count -- a cell is in it exactly when its value was computed.
    """
    memo: dict[tuple[int, int], int] = {}
    stack: list[tuple[int, int, int]] = [(d, t, 0)]
    while stack:
        cd, ct, phase = stack.pop()
        if (cd, ct) in memo:
            continue
        if phase == 0:
            b = _base(cd, ct)
            if b is not None:
                memo[(cd, ct)] = b
                continue
            # D_{d-2}(t-1) is always needed; D_{d-1}(t-1) decides the OR.
            stack.append((cd, ct, 1))
            for child in ((cd - 1, ct - 1), (cd - 2, ct - 1)):
                if child not in memo:
                    stack.append((child[0], child[1], 0))
        elif phase == 1:
            left = memo[(cd - 2, ct - 1)]
            mid = memo[(cd - 1, ct - 1)]
            if short_circuit and mid == 1:
                # OR is 1 regardless of D_d(t-1): that sub-cone is not needed.
                memo[(cd, ct)] = left ^ 1
                continue
            if (cd, ct - 1) in memo:
                memo[(cd, ct)] = left ^ (mid | memo[(cd, ct - 1)])
            else:
                stack.append((cd, ct, 2))
                stack.append((cd, ct - 1, 0))
        else:
            left = memo[(cd - 2, ct - 1)]
            mid = memo[(cd - 1, ct - 1)]
            memo[(cd, ct)] = left ^ (mid | memo[(cd, ct - 1)])
    return {"value": memo[(d, t)], "cells_evaluated": len(memo)}


def costs(d: int, t: int) -> dict:
    """All three cost models for the query D_d(t), plus the value."""
    pruned = trace(d, t, short_circuit=True)
    full = trace(d, t, short_circuit=False)
    if pruned["value"] != full["value"]:
        raise AssertionError(
            f"short-circuit changed the value at d={d}, t={t}: "
            f"{pruned['value']} vs {full['value']}")
    rect = (d + 1) * (t + 1)
    return {
        "d": d,
        "t": t,
        "value": int(pruned["value"]),
        "cells_rect": rect,
        "cells_cone": full["cells_evaluated"],
        "cells_evaluated": pruned["cells_evaluated"],
        "pruned_fraction": round(
            1.0 - pruned["cells_evaluated"] / full["cells_evaluated"], 6),
        "cone_over_rect": round(full["cells_evaluated"] / rect, 6),
    }


def growth(rows: list[dict]) -> dict:
    """Least-squares exponent of cells_evaluated against t, on log-log axes.

    An exponent near 2 means the saving is a constant factor on a quadratic
    cost. Anything meaningfully below 2 would be the interesting outcome.
    """
    pts = [(r["t"], r["cells_evaluated"]) for r in rows if r["t"] > 1]
    if len(pts) < 2:
        return {"exponent": None, "note": "need at least two depths"}
    xs = np.log2([p[0] for p in pts])
    ys = np.log2([p[1] for p in pts])
    slope, intercept = np.polyfit(xs, ys, 1)
    cone = np.polyfit(xs, np.log2([r["cells_cone"] for r in rows if r["t"] > 1]), 1)
    return {
        "exponent_evaluated": round(float(slope), 4),
        "exponent_cone": round(float(cone[0]), 4),
        "intercept_evaluated": round(float(intercept), 4),
        "reading": (
            "constant-factor saving on the same growth rate"
            if abs(float(slope) - float(cone[0])) < 0.05
            else "the short-circuit changes the growth rate -- investigate"),
    }


def oracle_check(rows: list[dict], steps: int) -> dict:
    """Every traced value against a straight simulation. Bit-exact or fail."""
    if not rows:
        return {"checked": 0, "ok": True}
    dg = reference_diagonals(steps, max(r["d"] for r in rows) + 1)
    bad = [{"d": r["d"], "t": r["t"], "traced": r["value"],
            "simulated": int(dg[r["t"], r["d"]])}
           for r in rows if int(dg[r["t"], r["d"]]) != r["value"]]
    return {"checked": len(rows), "mismatches": bad, "ok": not bad}


def run(depths: list[int]) -> dict:
    started = time.time()
    rows = [costs(t, t) for t in depths]          # the centre-bit query: d = t
    steps = max(depths)
    return {
        "artifact_type": ARTIFACT_TYPE,
        "artifact_version": ARTIFACT_VERSION,
        "query": "D_t(t) -- the diagonal the centre bit at time t lives on",
        "depths": depths,
        "rows": rows,
        "growth": growth(rows),
        "oracle": oracle_check(rows, steps),
        "elapsed_s": round(time.time() - started, 2),
    }


def verify(path: Path) -> int:
    art = json.loads(path.read_text(encoding="utf-8"))
    if art.get("artifact_type") != ARTIFACT_TYPE:
        print(f"not a {ARTIFACT_TYPE} artifact", file=sys.stderr)
        return 2
    fresh = run(art["depths"])
    ok = True
    for old, new in zip(art["rows"], fresh["rows"]):
        for key in ("value", "cells_rect", "cells_cone", "cells_evaluated"):
            if old[key] != new[key]:
                print(f"MISMATCH d={old['d']} t={old['t']} {key}: "
                      f"{old[key]} -> {new[key]}", file=sys.stderr)
                ok = False
    if not fresh["oracle"]["ok"]:
        print("oracle disagreed with the trace", file=sys.stderr)
        ok = False
    print(f"verify: {len(fresh['rows'])} rows recomputed, "
          f"{'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def self_test() -> int:
    """The trace must agree with simulation everywhere, not just on the query."""
    steps, diagonals = 60, 40
    dg = reference_diagonals(steps, diagonals)
    bad = 0
    for t in range(1, steps + 1, 7):
        for d in range(0, min(diagonals, 2 * t + 1), 3):
            for sc in (True, False):
                got = trace(d, t, short_circuit=sc)["value"]
                if got != int(dg[t, d]):
                    print(f"FAIL d={d} t={t} sc={sc}: {got} != {dg[t, d]}",
                          file=sys.stderr)
                    bad += 1
    # The short-circuit must never evaluate MORE cells than the full cone.
    for t in (20, 40, 60):
        c = costs(t, t)
        if c["cells_evaluated"] > c["cells_cone"]:
            print(f"FAIL pruning increased cost at t={t}", file=sys.stderr)
            bad += 1
    print(f"self-test: {'OK' if not bad else f'{bad} failure(s)'}")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depths", default="16,32,64,128,256",
                    help="comma-separated t values for the query D_t(t)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--verify", type=Path)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.verify:
        return verify(args.verify)

    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    art = run(depths)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(art, indent=2) + "\n", encoding="utf-8")
    if args.pretty:
        print(f"{'t':>7} {'rect':>12} {'cone':>12} {'pruned':>12} {'saved':>8}",
              file=sys.stderr)
        for r in art["rows"]:
            print(f"{r['t']:>7} {r['cells_rect']:>12,} {r['cells_cone']:>12,} "
                  f"{r['cells_evaluated']:>12,} {r['pruned_fraction']:>7.1%}",
                  file=sys.stderr)
        print(f"  growth: {art['growth']}", file=sys.stderr)
        print(f"  oracle: {art['oracle']['checked']} checked, "
              f"ok={art['oracle']['ok']}", file=sys.stderr)
    print(json.dumps(art, indent=2))
    return 0 if art["oracle"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

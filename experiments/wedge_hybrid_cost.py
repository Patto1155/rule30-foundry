"""How much of a centre-bit query does the settled wedge actually supply?

docs/theory/README.md §3 closes the wedge as a *shortcut*: the centre bit at
time T lives on diagonal d = T, and settle(T) ~ 1.34*T > T, so the target cell
is permanently unsettled. That argument is about the TARGET. It does not say
the target's dependency cone contains no settled cells -- cells at smaller d
inside the cone settle long before the horizon -- and the quantity that decides
whether the wedge is worth building on is the fraction of the cone it supplies,
which has never been measured.

This script measures it exactly, and checks the answer against the simulator.

    a(T, 0) = D_T(T)                                    the centre bit
    cone    = { (d,t) : the backward closure of D_T(T) under (R) }
    free    = { (d,t) in cone : t >= settle(d) }        O(1) from the pattern map
    work    = cone \\ free                               must be evaluated

`useful success` would be `work` growing more slowly than `cone`. `useful
failure` is `work` still quadratic, which makes further wedge optimisation a
weak route to centre bits and says so with a number rather than an argument.

Reported separately, because they are different claims: the constant-factor
saving (a practical speedup) and the growth exponent (an asymptotic one).

CAVEAT, stated because it bounds the result. settle(d) is measured inside a
finite horizon, so it is an upper bound on the true settling time: a diagonal
that looks settled here could in principle break later. That direction is the
safe one -- it can only make the wedge look MORE useful than it is, so a
negative result is not weakened by it.

Usage:
    python experiments/wedge_hybrid_cost.py --depths 128,256,512 --pretty
    python experiments/wedge_hybrid_cost.py --verify runs/<artifact>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ARTIFACT_TYPE = "rule30.wedge_hybrid_cost"
ARTIFACT_VERSION = 1
PERIODS = (1, 2, 4, 8, 16, 32, 64)


def left_diagonals(steps: int, diagonals: int) -> np.ndarray:
    """[t, d] = D_d(t) = a(t, -t+d), by direct simulation. The oracle."""
    width = 2 * steps + 3
    centre = steps + 1
    mask = (1 << width) - 1
    wmask = (1 << diagonals) - 1
    nbytes = (diagonals + 7) // 8
    out = np.empty((steps + 1, diagonals), dtype=np.uint8)
    state = 1 << centre
    for t in range(steps + 1):
        win = (state >> (centre - t)) & wmask
        out[t] = np.unpackbits(
            np.frombuffer(win.to_bytes(nbytes, "little"), dtype=np.uint8),
            bitorder="little", count=diagonals)
        state = ((state << 1) ^ (state | (state >> 1))) & mask
    return out


def settle_times(dg: np.ndarray, tail: int) -> np.ndarray:
    """settle[d] = earliest t from which column d is exactly p-periodic.

    np.iinfo(np.int64).max marks a diagonal that never settles in the horizon,
    so it is never counted as free.
    """
    n_t, n_d = dg.shape
    never = np.iinfo(np.int64).max
    out = np.full(n_d, never, dtype=np.int64)
    for d in range(n_d):
        col = dg[:, d]
        seg = col[-tail:] if n_t > tail else col
        period = None
        for p in PERIODS:
            if len(seg) > 2 * p and np.array_equal(seg[:-p], seg[p:]):
                period = p
                break
        if period is None:
            continue
        # Walk back while the p-shift still agrees.
        t = n_t - period - 1
        while t >= 0 and col[t] == col[t + period]:
            t -= 1
        out[d] = t + 1
    return out


def cone_cells(T: int) -> np.ndarray:
    """For each t in 0..T, the [lo, hi] band of d in the cone of D_T(T).

    One step back from (d, t+1) reaches d-2, d-1 and d at time t, so after k
    steps the band is [T - 2k, T], clipped at 0.
    """
    ts = np.arange(T + 1)
    lo = np.maximum(0, T - 2 * (T - ts))
    hi = np.full(T + 1, T)
    return np.stack([lo, hi], axis=1)


def hybrid(T: int, dg: np.ndarray, settle: np.ndarray) -> dict:
    """Exact cell accounting for the query D_T(T), plus the verified value."""
    band = cone_cells(T)
    lo, hi = band[:, 0], band[:, 1]
    widths = hi - lo + 1
    cone = int(widths.sum())
    # free[t] = how many d in [lo_t, hi_t] have settle[d] <= t.
    free = 0
    for t in range(T + 1):
        seg = settle[lo[t]:hi[t] + 1]
        free += int((seg <= t).sum())
    work = cone - free
    return {
        "T": T,
        "centre_bit": int(dg[T, T]),
        "cells_cone": cone,
        "cells_free": free,
        "cells_work": work,
        "free_fraction": round(free / cone, 6),
        "settle_at_T": (None if settle[T] == np.iinfo(np.int64).max
                        else int(settle[T])),
        "target_is_settled": bool(settle[T] <= T),
    }


def centre_column(T: int) -> int:
    """a(T,0) by plain simulation, sharing nothing with the diagonal path."""
    width = 2 * T + 3
    centre = T + 1
    mask = (1 << width) - 1
    state = 1 << centre
    for _ in range(T):
        state = ((state << 1) ^ (state | (state >> 1))) & mask
    return (state >> centre) & 1


def growth(rows: list[dict]) -> dict:
    pts = [r for r in rows if r["T"] > 1 and r["cells_work"] > 0]
    if len(pts) < 2:
        return {"exponent_work": None, "note": "need at least two depths"}
    xs = np.log2([r["T"] for r in pts])
    work = np.polyfit(xs, np.log2([r["cells_work"] for r in pts]), 1)[0]
    cone = np.polyfit(xs, np.log2([r["cells_cone"] for r in pts]), 1)[0]
    frac = np.polyfit(xs, np.log2([r["free_fraction"] for r in pts]), 1)[0]
    delta = float(work) - float(cone)
    if delta < -0.05:
        reading = ("work grows more slowly than the cone: the wedge changes "
                   "the growth rate, which is the outcome worth chasing")
    elif delta > 0.05:
        reading = ("work grows FASTER than the cone: the free fraction decays, "
                   "so the wedge contributes relatively less as T grows")
    else:
        reading = "the wedge buys a constant factor, not a growth rate"
    return {
        "exponent_work": round(float(work), 4),
        "exponent_cone": round(float(cone), 4),
        "exponent_free_fraction": round(float(frac), 4),
        "free_fraction_first": pts[0]["free_fraction"],
        "free_fraction_last": pts[-1]["free_fraction"],
        "reading": reading,
    }


def run(depths: list[int], tail: int) -> dict:
    started = time.time()
    T = max(depths)
    dg = left_diagonals(T, T + 1)
    settle = settle_times(dg, tail)
    rows = [hybrid(t, dg, settle) for t in depths]
    oracle = [{"T": r["T"], "hybrid": r["centre_bit"], "simulated": centre_column(r["T"])}
              for r in rows]
    bad = [o for o in oracle if o["hybrid"] != o["simulated"]]
    return {
        "artifact_type": ARTIFACT_TYPE,
        "artifact_version": ARTIFACT_VERSION,
        "params": {"depths": depths, "tail": tail},
        "rows": rows,
        "growth": growth(rows),
        "oracle": {"checked": len(oracle), "mismatches": bad, "ok": not bad},
        "target_ever_settled": any(r["target_is_settled"] for r in rows),
        "elapsed_s": round(time.time() - started, 2),
    }


def verify(path: Path) -> int:
    art = json.loads(path.read_text(encoding="utf-8"))
    if art.get("artifact_type") != ARTIFACT_TYPE:
        print(f"not a {ARTIFACT_TYPE} artifact", file=sys.stderr)
        return 2
    fresh = run(art["params"]["depths"], art["params"]["tail"])
    ok = True
    for old, new in zip(art["rows"], fresh["rows"]):
        for key in ("centre_bit", "cells_cone", "cells_free", "cells_work"):
            if old[key] != new[key]:
                print(f"MISMATCH T={old['T']} {key}: {old[key]} -> {new[key]}",
                      file=sys.stderr)
                ok = False
    if not fresh["oracle"]["ok"]:
        print("oracle disagreed", file=sys.stderr)
        ok = False
    print(f"verify: {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def self_test() -> int:
    bad = 0
    dg = left_diagonals(64, 65)
    for T in (8, 16, 32, 64):
        if int(dg[T, T]) != centre_column(T):
            print(f"FAIL centre bit at T={T}", file=sys.stderr)
            bad += 1
    # D_0 == 1, D_1(t>=1) == 1: the cone edge, from docs/theory/README.md §3.
    if not dg[:, 0].all() or not dg[1:, 1].all():
        print("FAIL cone-edge base cases", file=sys.stderr)
        bad += 1
    # The cone band must never exceed the full rectangle.
    band = cone_cells(32)
    if int((band[:, 1] - band[:, 0] + 1).sum()) > 33 * 33:
        print("FAIL cone larger than its rectangle", file=sys.stderr)
        bad += 1
    print(f"self-test: {'OK' if not bad else f'{bad} failure(s)'}")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depths", default="64,128,256,512")
    ap.add_argument("--tail", type=int, default=256)
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
    art = run(depths, args.tail)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(art, indent=2) + "\n", encoding="utf-8")
    if args.pretty:
        print(f"{'T':>6} {'cone':>12} {'free':>12} {'work':>12} {'free%':>7} "
              f"{'settle(T)':>10} {'target settled':>15}", file=sys.stderr)
        for r in art["rows"]:
            print(f"{r['T']:>6} {r['cells_cone']:>12,} {r['cells_free']:>12,} "
                  f"{r['cells_work']:>12,} {r['free_fraction']:>6.1%} "
                  f"{str(r['settle_at_T']):>10} {str(r['target_is_settled']):>15}",
                  file=sys.stderr)
        print(f"  growth: {art['growth']}", file=sys.stderr)
        print(f"  oracle: {art['oracle']['checked']} centre bits, "
              f"ok={art['oracle']['ok']}", file=sys.stderr)
    print(json.dumps(art, indent=2))
    return 0 if art["oracle"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

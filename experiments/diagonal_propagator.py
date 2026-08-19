"""Rolling 3-diagonal propagator for the left diagonals, with O(T) memory.

WHY. left_diagonals() simulates the whole cone and materializes a
(steps+1, diagonals) array -- O(d*T) memory, 312 MB at T=26000, d=12000, and
hopeless at the d ~ 10^6 needed to decide period-16. But the diagonal recursion

    D_d(t+1) = D_{d-2}(t) XOR ( D_{d-1}(t) OR D_d(t) )                    (R)

is exact for the ACTUAL sequences, not merely their eventual patterns, so
diagonals can be propagated with a rolling window of three: O(T) memory,
independent of d.

HOW IT VECTORIZES. (R) self-resets. Writing a = D_{d-2}, b = D_{d-1}, x = D_d:

    b[t] = 1  ->  x[t+1] = NOT a[t]          (reset: x[t] is irrelevant)
    b[t] = 0  ->  x[t+1] = a[t] XOR x[t]     (propagate)

So with P the prefix XOR of a (P[0] = 0, P[t+1] = P[t] XOR a[t]) and r[t] the
last reset index <= t:

    r[t] >= 0  ->  x[t+1] = (1 XOR a[r]) XOR P[t+1] XOR P[r+1]
    r[t] <  0  ->  x[t+1] = x[0] XOR P[t+1]

which is a segmented prefix-XOR: one np.bitwise_xor.accumulate and one
np.maximum.accumulate per diagonal, no serial t-loop. Since diagonals are
roughly half ones, resets are ~2 apart and segments are short -- but the closed
form above does not depend on that, it is exact for any segment length.

NOTE the connection to the zero-word branch points. A reset exists in the
settled region iff the settled word of d-1 is nonzero. When w_{d-1} = 0 there is
NO reset in the period, the scan must reach back past the transient to the last
1 in D_{d-1}, and that is precisely why the 16-bit pattern map goes ambiguous
there. This propagator resolves such a branch point because it carries the
actual sequence, not just the eventual word.

Seeds: D_0(t) = 1 for all t, D_1(0) = 0 and D_1(t) = 1 for t >= 1, and
D_d(0) = 0 for d >= 1 (single seed at cell 0).

Run:  python experiments/diagonal_propagator.py --pretty
Exits non-zero if the propagator disagrees with left_diagonals anywhere.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagonal_recursion import left_diagonals  # noqa: E402

PERIOD = 16


def step_diagonal(a: np.ndarray, b: np.ndarray, x0: int = 0) -> np.ndarray:
    """One diagonal from the previous two, as a segmented prefix-XOR scan.

    a = D_{d-2}, b = D_{d-1}, both length n over t = 0..n-1. Returns x = D_d of
    the same length with x[0] = x0 and x[t+1] = a[t] XOR (b[t] OR x[t]).
    """
    n = a.size
    x = np.empty(n, dtype=np.uint8)
    x[0] = x0
    if n == 1:
        return x

    am = a[: n - 1]
    bm = b[: n - 1]

    # P[t] = XOR of a[0..t-1]
    P = np.empty(n, dtype=np.uint8)
    P[0] = 0
    np.bitwise_xor.accumulate(am, out=P[1:])

    # r[t] = last index j <= t with b[j] == 1, else -1
    idx = np.arange(n - 1, dtype=np.int64)
    r = np.maximum.accumulate(np.where(bm == 1, idx, np.int64(-1)))

    has = r >= 0
    rr = np.where(has, r, 0)
    reset_val = (a[rr] ^ np.uint8(1)) ^ P[1:] ^ P[rr + 1]
    plain_val = np.uint8(x0) ^ P[1:]
    x[1:] = np.where(has, reset_val, plain_val)
    return x


def step_diagonal_naive(a: np.ndarray, b: np.ndarray, x0: int = 0) -> np.ndarray:
    """Serial reference for (R). Only for correctness and benchmark contrast."""
    n = a.size
    x = np.empty(n, dtype=np.uint8)
    x[0] = x0
    for t in range(n - 1):
        x[t + 1] = a[t] ^ (b[t] | x[t])
    return x


def seed_diagonals(n: int) -> tuple[np.ndarray, np.ndarray]:
    """D_0 and D_1 over t = 0..n-1."""
    d0 = np.ones(n, dtype=np.uint8)
    d1 = np.ones(n, dtype=np.uint8)
    d1[0] = 0
    return d0, d1


def settled_word(col: np.ndarray, tail: int, period: int = PERIOD):
    """The period-locked word of a diagonal, or None if not settled in the tail."""
    n = col.size
    base = max(0, n - tail)
    if base % period:
        raise ValueError(f"tail must leave base ({base}) a multiple of {period}")
    seg = col[base:]
    if seg.size <= 2 * period or not np.array_equal(seg[:-period], seg[period:]):
        return None
    w = 0
    for i in range(period):
        if seg[i]:
            w |= 1 << ((base + i) % period)
    return w


def verify_against_simulation(steps: int, diagonals: int) -> dict:
    """np.array_equal on EVERY diagonal, not a sample."""
    dg = left_diagonals(steps, diagonals)
    n = steps + 1
    prev2, prev1 = seed_diagonals(n)

    mismatched = []
    if not np.array_equal(prev2, dg[:, 0]):
        mismatched.append(0)
    if diagonals > 1 and not np.array_equal(prev1, dg[:, 1]):
        mismatched.append(1)

    for d in range(2, diagonals):
        cur = step_diagonal(prev2, prev1, x0=0)
        if not np.array_equal(cur, dg[:, d]):
            mismatched.append(d)
            if len(mismatched) > 8:
                break
        prev2, prev1 = prev1, cur

    return {
        "steps": steps,
        "diagonals": diagonals,
        "compared_cells": int(n) * int(diagonals),
        "method": "np.array_equal per diagonal, all diagonals",
        "mismatched_diagonals": mismatched,
        "ok": not mismatched,
    }


def bench(steps: int, diagonals: int, repeats: int = 3) -> dict:
    """Honest timing: same work both ways, repeated, with run-to-run spread."""
    n = steps + 1

    base_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        dg = left_diagonals(steps, diagonals)
        base_times.append(time.perf_counter() - t0)
    baseline_bytes = int(dg.nbytes)
    del dg

    prop_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        prev2, prev1 = seed_diagonals(n)
        for _d in range(2, diagonals):
            cur = step_diagonal(prev2, prev1, x0=0)
            prev2, prev1 = prev1, cur
        prop_times.append(time.perf_counter() - t0)
    # rolling window (3) + prefix + index array (int64) + a couple of temporaries
    prop_bytes = int(3 * n + 2 * n + 8 * n)

    # vectorized vs serial, on ONE diagonal, to evidence the vectorization claim
    a, b = seed_diagonals(n)
    t0 = time.perf_counter()
    xv = step_diagonal(a, b)
    vec_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    xn = step_diagonal_naive(a, b)
    naive_t = time.perf_counter() - t0

    def summarize(ts):
        return {"best": round(min(ts), 3), "worst": round(max(ts), 3),
                "mean": round(sum(ts) / len(ts), 3),
                "spread_pct": round(100 * (max(ts) - min(ts)) / min(ts), 1)}

    return {
        "repeats": repeats,
        "baseline_left_diagonals_s": summarize(base_times),
        "propagator_s": summarize(prop_times),
        "speedup_on_mean": round((sum(base_times) / len(base_times))
                                 / (sum(prop_times) / len(prop_times)), 2),
        "baseline_peak_bytes": baseline_bytes,
        "propagator_working_bytes": prop_bytes,
        "memory_ratio": round(baseline_bytes / prop_bytes, 1),
        "single_diagonal": {
            "vectorized_s": round(vec_t, 4),
            "serial_s": round(naive_t, 4),
            "speedup": round(naive_t / vec_t, 1) if vec_t > 0 else None,
            "agree": bool(np.array_equal(xv, xn)),
        },
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=26000)
    p.add_argument("--diagonals", type=int, default=12000)
    p.add_argument("--bench-steps", type=int, default=8000)
    p.add_argument("--bench-diagonals", type=int, default=4000)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--skip-bench", action="store_true")
    p.add_argument("--pretty", action="store_true")
    p.add_argument("--out")
    a = p.parse_args(argv)

    t0 = time.time()
    ver = verify_against_simulation(a.steps, a.diagonals)
    bm = None if a.skip_bench else bench(a.bench_steps, a.bench_diagonals, a.repeats)

    out = {
        "artifact_type": "rule30.diagonal_propagator",
        "artifact_version": 1,
        "verification": ver,
        "benchmark": bm,
        "elapsed_s": round(time.time() - t0, 3),
        "ok": ver["ok"],
    }

    if a.pretty:
        w = sys.stderr
        print(f"verification : {'PASS' if ver['ok'] else 'FAIL'}  "
              f"{ver['compared_cells']:,} cells, all {ver['diagonals']} diagonals "
              f"by np.array_equal", file=w)
        if ver["mismatched_diagonals"]:
            print(f"  mismatched : {ver['mismatched_diagonals']}", file=w)
        if bm:
            b = bm
            print("", file=w)
            print(f"benchmark (T={a.bench_steps}, d={a.bench_diagonals}, "
                  f"{b['repeats']} repeats):", file=w)
            for label, key in (("left_diagonals", "baseline_left_diagonals_s"),
                               ("propagator", "propagator_s")):
                s = b[key]
                print(f"  {label:<16} best {s['best']:>7.3f}s  mean {s['mean']:>7.3f}s"
                      f"  worst {s['worst']:>7.3f}s  (spread {s['spread_pct']}%)",
                      file=w)
            print(f"  speedup (mean)   {b['speedup_on_mean']}x", file=w)
            print(f"  memory           {b['baseline_peak_bytes']/1e6:.1f} MB -> "
                  f"{b['propagator_working_bytes']/1e6:.1f} MB "
                  f"({b['memory_ratio']}x less)", file=w)
            sd = b["single_diagonal"]
            print(f"  one diagonal     vectorized {sd['vectorized_s']}s vs serial "
                  f"{sd['serial_s']}s = {sd['speedup']}x, agree={sd['agree']}",
                  file=w)

    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=str),
                               encoding="utf-8")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Independent replication: left-diagonal settling law of open-boundary Rule 30.

INDEPENDENT CODE PATH.  Nothing here was copied from the original measurement.
The simulator used for the big run is a *moving-frame* rewrite of Rule 30 that is
bit-verified, before any measurement, against the repo's trusted reference
(`experiments/rule30_open_utils.py`, whose packed CPU path is itself checked
against a naive cell-by-cell Rule 30).

Object under test
-----------------
Rule 30, single black cell at position 0, open boundaries, position increasing
to the RIGHT:

    a(t+1, i) = a(t, i-1) XOR ( a(t, i) OR a(t, i+1) )

Left diagonal d is  D_d(t) = a(t, -t + d)  (offset d inward from the leftmost
cell of the light cone, which sits at position -t).

Moving-frame identity (the whole trick)
---------------------------------------
Put  b(t, d) := a(t, -t + d).  Then

    b(t+1, d) = a(t+1, -(t+1)+d)
              = a(t, -t-2+d) XOR ( a(t, -t-1+d) OR a(t, -t+d) )
              = b(t, d-2)     XOR ( b(t, d-1)   OR b(t, d)   )

so in the moving frame the update reads ONLY at offsets d-2, d-1, d.  Two
consequences that matter:

  * There is **no rightward dependence at all**, so a truncated array of width W
    is EXACT for every d < W and every t -- no boundary slack needed, no light
    cone to pad.  (This is the independent-path analogue of the fact that
    a(t,-t+d) depends only on initial cells in [d-2t, d].)
  * b(t,-1) = b(t,-2) = 0 identically (those are outside the light cone), so the
    left edge of the packed buffer needs no special case.

In packed little-bit-order form one step is:  out = (c<<2) ^ ((c<<1) | c).

Usage
-----
    PYTHONUTF8=1 python experiments/wedge_verify.py --test          # gates only, fast
    PYTHONUTF8=1 python experiments/wedge_verify.py --steps 65536   # full run
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

A051023_PREFIX = "1101110011000101"  # Rule 30 center column, OEIS A051023

# Candidate periods from the claim's specification.
SPEC_PERIODS = (1, 2, 3, 4, 6, 8, 12, 16, 32, 64)


# --------------------------------------------------------------------------
# Moving-frame Rule 30 (independent simulator)
# --------------------------------------------------------------------------


def frame_step_naive(b: np.ndarray) -> np.ndarray:
    """Cell-by-cell moving-frame step; the ground truth for the packed version."""
    out = np.zeros_like(b)
    for d in range(len(b)):
        m2 = b[d - 2] if d >= 2 else 0
        m1 = b[d - 1] if d >= 1 else 0
        m0 = b[d]
        out[d] = m2 ^ (m1 | m0)
    return out


def frame_step_packed(cur: np.ndarray) -> np.ndarray:
    """Packed uint64 moving-frame step (little bit order: cell d -> word d//64, bit d%64)."""
    s1 = cur << np.uint64(1)
    s1[1:] |= cur[:-1] >> np.uint64(63)
    s2 = cur << np.uint64(2)
    s2[1:] |= cur[:-1] >> np.uint64(62)
    return s2 ^ (s1 | cur)


def frame_seed(width_words: int) -> np.ndarray:
    """b(0, d) = a(0, d) = [d == 0]."""
    cur = np.zeros(width_words, dtype=np.uint64)
    cur[0] = np.uint64(1)
    return cur


def frame_field(n_steps: int, width: int) -> np.ndarray:
    """Dense uint8 field b[t, d] for t in 0..n_steps, d in 0..width-1 (small runs only)."""
    if width % 64:
        raise ValueError("width must be a multiple of 64")
    nw = width // 64
    cur = frame_seed(nw)
    out = np.zeros((n_steps + 1, width), dtype=np.uint8)
    for t in range(n_steps + 1):
        out[t] = unpack_row(cur, width)
        if t < n_steps:
            cur = frame_step_packed(cur)
    return out


def unpack_row(packed: np.ndarray, width: int) -> np.ndarray:
    return np.unpackbits(packed.view(np.uint8), bitorder="little")[:width]


# --------------------------------------------------------------------------
# Verification gates (MUST all pass before any measurement)
# --------------------------------------------------------------------------


def gate_repo_self_tests() -> dict:
    import rule30_open_utils as U

    U.verify_single_spike_direction_and_boundary()
    U.verify_random_batch_against_naive()
    U.verify_spacetime_against_naive()
    return {"repo_self_tests": "PASS", "gpu_available": bool(U.GPU_AVAILABLE)}


def gate_packed_vs_naive_frame(seed: int = 20260815) -> dict:
    """My packed moving-frame step == my naive moving-frame step, incl. random ICs.

    Random ICs are mandatory here: the repo already paid for the lesson that a
    lone-spike test hides word-boundary / edge bugs (docs/GPU_KERNELS.md).
    """
    rng = np.random.default_rng(seed)
    width = 256  # 4 uint64 words -> exercises inter-word carries
    for trial in range(64):
        if trial == 0:
            row = np.zeros(width, dtype=np.uint8)
            row[0] = 1
        elif trial == 1:
            row = np.ones(width, dtype=np.uint8)
        else:
            row = rng.integers(0, 2, size=width, dtype=np.uint8)
        packed = np.packbits(row, bitorder="little").view(np.uint64).copy()
        for _ in range(40):
            naive = frame_step_naive(row)
            packed = frame_step_packed(packed)
            row = naive
            if not np.array_equal(unpack_row(packed, width), row):
                raise RuntimeError(f"packed frame step != naive frame step (trial {trial})")
    return {"packed_vs_naive_frame": "PASS"}


def gate_frame_vs_repo_spacetime(n_steps: int = 1500, width: int = 1216) -> dict:
    """My moving-frame diagonals == diagonals read out of the repo's own spacetime field.

    This is the load-bearing gate: it ties my index convention to the repo's
    trusted lab-frame simulator, so an off-by-one in "position -t+d at time t"
    cannot survive.
    """
    import rule30_open_utils as U

    # Lab-frame tape. Seed at array index `center`; array index = center + position.
    # Right truncation is exact: a(t,-t+d) depends only on initial cells <= d.
    center = n_steps + 2
    n_cells = center + width + 4
    row0 = U.make_single_spike_row(n_cells, center)
    field = U.simulate_spacetime(row0, n_steps + 1, gpu=False)  # times 0 .. n_steps

    mine = frame_field(n_steps, width)
    theirs = np.zeros_like(mine)
    for t in range(n_steps + 1):
        theirs[t] = field[t, center - t : center - t + width]

    if not np.array_equal(mine, theirs):
        bad = np.argwhere(mine != theirs)
        raise RuntimeError(f"moving frame != repo spacetime; first mismatch (t,d)={tuple(bad[0])}")

    return {
        "frame_vs_repo_spacetime": "PASS",
        "gate_steps": n_steps,
        "gate_width": width,
        "cells_compared": int(mine.size),
    }


def gate_center_column(n_steps: int = 512) -> dict:
    """a(t,0) = b(t,t): center column must be OEIS A051023, and must match the repo's."""
    import rule30_open_utils as U

    width = 64 * ((n_steps + 64) // 64 + 1)
    field = frame_field(n_steps, width)
    mine = np.array([field[t, t] for t in range(n_steps + 1)], dtype=np.uint8)
    prefix = "".join(str(int(v)) for v in mine[:16])
    if prefix != A051023_PREFIX:
        raise RuntimeError(f"center column prefix {prefix!r} != A051023 {A051023_PREFIX!r}")

    n_cells = 2 * n_steps + 65
    center = n_steps + 32
    row0 = U.make_single_spike_row(n_cells, center)
    repo_col = U.simulate_center_columns_batch(row0, n_steps, center, gpu=False)[0]
    if not np.array_equal(mine, repo_col):
        raise RuntimeError("center column from moving frame != repo center column")

    return {"center_column_A051023": "PASS", "center_prefix": prefix, "center_steps": n_steps}


def gate_edge_diagonals(n_steps: int = 4096) -> dict:
    """d=0 and d=1 must be identically 1 for t >= 1 (the spec's indexing check)."""
    width = 128
    field = frame_field(n_steps, width)
    d0 = field[:, 0]
    d1 = field[:, 1]
    ok0 = bool(np.all(d0[1:] == 1))
    ok1 = bool(np.all(d1[1:] == 1))
    if not (ok0 and ok1):
        raise RuntimeError(f"edge diagonals not all-ones: d0={ok0} d1={ok1}")
    return {
        "edge_diagonals_all_ones": "PASS",
        "d0_t0": int(d0[0]),
        "d1_t0": int(d1[0]),  # note: 0 at t=0 -- the claim holds only for t>=1
    }


def run_gates(verbose: bool = True) -> dict:
    gates = {}
    for fn in (
        gate_repo_self_tests,
        gate_packed_vs_naive_frame,
        gate_center_column,
        gate_edge_diagonals,
        gate_frame_vs_repo_spacetime,
    ):
        t0 = time.time()
        res = fn()
        gates.update(res)
        if verbose:
            print(f"[gate] {fn.__name__:32s} ok ({time.time() - t0:.1f}s)", file=sys.stderr)
    return gates


# --------------------------------------------------------------------------
# Main measurement pass (streaming; O(P*W) memory, not O(T*W))
# --------------------------------------------------------------------------


def measure(n_steps: int, width: int, periods=SPEC_PERIODS, n_sample: int = 256, verbose: bool = True):
    """Return (last_mismatch[len(periods), width] int32, sampled columns, sample_d).

    last_mismatch[i, d] = max{ t <= n_steps : D_d(t) != D_d(t - periods[i]) }, else 0.
    """
    if sys.byteorder != "little":
        raise RuntimeError("packed bit indexing assumes a little-endian host")
    if width % 64:
        raise ValueError("width must be a multiple of 64")
    nw = width // 64
    periods = tuple(int(p) for p in periods)
    pmax = max(periods)

    ring = np.zeros((pmax + 1, nw), dtype=np.uint64)
    cur = frame_seed(nw)
    ring[0] = cur

    last = np.zeros((len(periods), width), dtype=np.int32)

    sample_d = np.unique(np.linspace(2, width - 1, n_sample).astype(np.int64))
    s_word = (sample_d // 64).astype(np.int64)
    s_bit = (sample_d % 64).astype(np.uint64)
    cols = np.zeros((n_steps + 1, sample_d.size), dtype=np.uint8)
    cols[0] = ((cur[s_word] >> s_bit) & np.uint64(1)).astype(np.uint8)

    t_start = time.time()
    for t in range(1, n_steps + 1):
        cur = frame_step_packed(cur)
        ring[t % (pmax + 1)] = cur
        cols[t] = ((cur[s_word] >> s_bit) & np.uint64(1)).astype(np.uint8)

        for i, p in enumerate(periods):
            if t < p:
                continue
            x = cur ^ ring[(t - p) % (pmax + 1)]
            nz = np.flatnonzero(x)
            if nz.size == 0:
                continue
            w0 = int(nz[0])
            w1 = int(nz[-1]) + 1
            bits = np.unpackbits(x[w0:w1].view(np.uint8), bitorder="little")
            idx = np.flatnonzero(bits)
            last[i, idx + w0 * 64] = t

        if verbose and t % 8192 == 0:
            print(f"  [sim] t={t}/{n_steps}  {time.time() - t_start:.0f}s", file=sys.stderr)

    return last, cols, sample_d


def settle_from_last(last: np.ndarray, periods=SPEC_PERIODS):
    """settle(d) = min_p last_mismatch(d,p); also return the argmin period."""
    settle = last.min(axis=0)
    best_p = np.asarray(periods, dtype=np.int32)[last.argmin(axis=0)]
    return settle, best_p


# --------------------------------------------------------------------------
# Period-cap insensitivity (Claim 2), on sampled full columns
# --------------------------------------------------------------------------


def last_mismatch_series(x: np.ndarray, p: int) -> int:
    """max{t : x[t] != x[t-p]} over t in [p, len(x)-1]; 0 if none."""
    if p >= x.size:
        return int(x.size)
    diff = x[p:] != x[:-p]
    if not diff.any():
        return 0
    # last True index in `diff` corresponds to t = p + that index
    return int(p + diff.size - 1 - int(np.argmax(diff[::-1])))


def cap_study(cols: np.ndarray, sample_d: np.ndarray, caps=(16, 64, 256, 1024), n_diag: int = 64,
              verbose: bool = True) -> dict:
    """settle_C(d) = min_{p<=C} last_mismatch(d,p) for C in caps, EXHAUSTIVE over p<=C."""
    take = np.unique(np.linspace(0, sample_d.size - 1, n_diag).astype(np.int64))
    cmax = max(caps)
    rows = []
    t0 = time.time()
    for j, si in enumerate(take):
        d = int(sample_d[si])
        x = cols[:, si]
        lm = np.array([last_mismatch_series(x, p) for p in range(1, cmax + 1)], dtype=np.int64)
        rec = {"d": d}
        for c in caps:
            sub = lm[:c]
            rec[f"settle_cap{c}"] = int(sub.min())
            rec[f"argmin_p_cap{c}"] = int(sub.argmin() + 1)
        # minimal p for which the trailing 4096 samples are exactly p-periodic
        tail = x[-4096:]
        p0 = None
        for p in range(1, min(cmax, tail.size // 2) + 1):
            if np.array_equal(tail[p:], tail[:-p]):
                p0 = p
                break
        rec["min_tail_period"] = p0
        rows.append(rec)
        if verbose and (j + 1) % 16 == 0:
            print(f"  [cap] {j + 1}/{take.size} diagonals  {time.time() - t0:.0f}s", file=sys.stderr)
    return {"caps": list(caps), "rows": rows}


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def fit_block(d: np.ndarray, s: np.ndarray) -> dict:
    d = d.astype(np.float64)
    s = s.astype(np.float64)
    slope_origin = float((d * s).sum() / (d * d).sum())
    A = np.vstack([d, np.ones_like(d)]).T
    coef, *_ = np.linalg.lstsq(A, s, rcond=None)
    ratios = s / d
    resid = s - slope_origin * d
    return {
        "n": int(d.size),
        "d_lo": int(d.min()),
        "d_hi": int(d.max()),
        "slope_through_origin": slope_origin,
        "slope_with_intercept": float(coef[0]),
        "intercept": float(coef[1]),
        "ratio_mean": float(ratios.mean()),
        "ratio_median": float(np.median(ratios)),
        "resid_std": float(resid.std()),
    }


def analyse(settle: np.ndarray, best_p: np.ndarray, n_steps: int, d_min: int, d_fit_max: int,
            n_blocks: int = 12) -> dict:
    d_all = np.arange(settle.size)
    mask = (d_all >= d_min) & (d_all <= d_fit_max)
    d = d_all[mask]
    s = settle[mask]

    censor_thresh = 0.85 * n_steps
    n_censored = int((s > censor_thresh).sum())

    global_fit = fit_block(d, s)
    edges = np.linspace(d_min, d_fit_max, n_blocks + 1).astype(np.int64)
    blocks = []
    for k in range(n_blocks):
        lo, hi = edges[k], edges[k + 1]
        sel = (d >= lo) & (d < hi if k < n_blocks - 1 else d <= hi)
        if sel.sum() > 10:
            blocks.append(fit_block(d[sel], s[sel]))

    bslopes = np.array([b["slope_through_origin"] for b in blocks])
    bratios = np.array([b["ratio_median"] for b in blocks])

    pv, pc = np.unique(best_p[mask], return_counts=True)

    return {
        "d_min": int(d_min),
        "d_fit_max": int(d_fit_max),
        "n_diagonals_fit": int(d.size),
        "n_censored_in_fit": n_censored,
        "global": global_fit,
        "blocks": blocks,
        "block_slope_min": float(bslopes.min()),
        "block_slope_max": float(bslopes.max()),
        "block_slope_spread": float(bslopes.max() - bslopes.min()),
        "block_ratio_median_min": float(bratios.min()),
        "block_ratio_median_max": float(bratios.max()),
        "best_period_histogram": {int(k): int(v) for k, v in zip(pv, pc)},
    }


def claim4(settle: np.ndarray, n_steps: int, s_hat: float, alphas=(0.25, 0.5, 0.75, 0.9, 0.95)) -> list:
    """First diagonal whose settle time reaches into the trailing window [alpha*T, T]."""
    out = []
    for a in alphas:
        thresh = a * n_steps
        bad = np.flatnonzero(settle >= thresh)
        d_first = int(bad[0]) if bad.size else None
        pred = thresh / s_hat
        out.append(
            {
                "alpha": a,
                "threshold_t": thresh,
                "d_first_fail": d_first,
                "predicted": pred,
                "ratio_measured_over_predicted": (d_first / pred) if d_first else None,
            }
        )
    return out


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=65536, help="T (max time)")
    ap.add_argument("--width", type=int, default=None, help="W diagonals tracked (default ~0.75*T, x64)")
    ap.add_argument("--test", action="store_true", help="gates only + tiny run")
    ap.add_argument("--gates-only", action="store_true")
    ap.add_argument("--n-sample", type=int, default=256)
    ap.add_argument("--cap-diagonals", type=int, default=64)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--npy", type=str, default=None, help="dump the full settle(d) array")
    args = ap.parse_args()

    if args.test:
        args.steps = 8192

    result = {"tool": "wedge_verify", "steps": args.steps}
    result.update(run_gates())
    if args.gates_only:
        print(json.dumps(result, indent=2))
        return

    T = args.steps
    W = args.width if args.width else 64 * int(round(0.75 * T / 64))
    result["width"] = W

    print(f"[run] T={T} W={W} periods={SPEC_PERIODS}", file=sys.stderr)
    t0 = time.time()
    last, cols, sample_d = measure(T, W, SPEC_PERIODS, n_sample=args.n_sample)
    result["sim_seconds"] = round(time.time() - t0, 1)

    settle, best_p = settle_from_last(last, SPEC_PERIODS)
    if args.npy:
        np.savez_compressed(args.npy, settle=settle, best_p=best_p, last=last,
                            periods=np.array(SPEC_PERIODS))

    # Sanity: the two edge diagonals must settle immediately.
    result["settle_d0"] = int(settle[0])
    result["settle_d1"] = int(settle[1])

    # Fit range: stay well clear of the T-truncation censor.
    d_min = max(64, W // 64)
    d_fit_max = int(0.45 * T)
    d_fit_max = min(d_fit_max, W - 1)
    result["fit"] = analyse(settle, best_p, T, d_min, d_fit_max)

    s_hat = result["fit"]["global"]["slope_through_origin"]
    result["s_hat"] = s_hat
    result["derived"] = {
        "wedge_area_fraction_1_over_2s": 1.0 / (2 * s_hat),
        "inner_boundary_speed_1_minus_1_over_s": 1.0 - 1.0 / s_hat,
        "lambda_L_repo": 0.244,
        "difference_vs_lambda_L": (1.0 - 1.0 / s_hat) - 0.244,
    }
    result["claim4"] = claim4(settle, T, s_hat)

    print("[run] period-cap study...", file=sys.stderr)
    result["cap_study"] = cap_study(cols, sample_d, n_diag=args.cap_diagonals)

    # compact per-cap agreement summary
    rows = result["cap_study"]["rows"]
    caps = result["cap_study"]["caps"]
    base = caps[1] if len(caps) > 1 else caps[0]
    agree = {}
    for c in caps:
        same = sum(1 for r in rows if r[f"settle_cap{c}"] == r[f"settle_cap{base}"])
        agree[f"cap{c}_matches_cap{base}"] = f"{same}/{len(rows)}"
    result["cap_study"]["agreement"] = agree
    result["cap_study"]["disagreements"] = [
        {k: r[k] for k in ("d", "min_tail_period") + tuple(f"settle_cap{c}" for c in caps)
         + tuple(f"argmin_p_cap{c}" for c in caps)}
        for r in rows
        if len({r[f"settle_cap{c}"] for c in caps}) > 1
    ]
    result["cap_study"]["min_tail_period_hist"] = {
        str(k): sum(1 for r in rows if r["min_tail_period"] == k)
        for k in sorted({r["min_tail_period"] for r in rows}, key=lambda z: (z is None, z))
    }

    # Curve of settle vs d, decimated, for eyeballing linearity.
    step = max(1, W // 128)
    result["settle_curve"] = [[int(d), int(settle[d])] for d in range(0, W, step)]

    out = json.dumps(result, indent=2, default=str)
    if args.json:
        Path(args.json).write_text(out, encoding="utf-8")
        print(f"[run] wrote {args.json}", file=sys.stderr)
    else:
        print(out)


if __name__ == "__main__":
    main()

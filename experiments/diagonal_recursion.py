"""Left-diagonal recursion for Rule 30, and the period-16 conjecture.

With D_d(t) = a(t, -t+d) (the cell at fixed offset d inward from the LEFT edge
of the single-seed light cone), substituting i = -t+d-1 into the Rule 30 update

    a(t+1,i) = a(t,i-1) XOR ( a(t,i) OR a(t,i+1) )

yields a CLOSED recursion on the left diagonals alone:

    D_d(t+1) = D_{d-2}(t) XOR ( D_{d-1}(t) OR D_d(t) )          (R)

LEMMA (period propagation). If D_{d-2} and D_{d-1} are eventually periodic with
common period p, then D_d is eventually periodic with period dividing 2p.
  Proof: phi_t(x) = D_{d-2}(t) XOR (D_{d-1}(t) OR x) is constant when
  D_{d-1}(t)=1 and the bijection x -> D_{d-2}(t) XOR x when D_{d-1}(t)=0. The
  one-period composite is therefore constant or x -> x XOR c. If any t in the
  period has D_{d-1}(t)=1 the composite is constant and D_d is eventually
  p-periodic; otherwise D_{d-1} == 0 on the period and the period is p or 2p. []

COROLLARY. D_0 == 1 (t>=0) and D_1 == 1 (t>=1) follow directly from the rule at
the cone edge, so by induction EVERY left diagonal is eventually periodic.

CONJECTURE (period-16). Every left diagonal is eventually periodic with period
dividing 16. This script is the verifier: it checks (R) bit-exactly and searches
for any settled diagonal whose period exceeds 16.

Usage:
    python experiments/diagonal_recursion.py --steps 20000 --diagonals 3000 --pretty
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter

import numpy as np

ARTIFACT_VERSION = 1
OEIS_A051023_PREFIX = "1101110011000101"
PERIOD_CANDIDATES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def left_diagonals(steps: int, diagonals: int) -> np.ndarray:
    """(steps+1, diagonals) uint8 array with entry [t, d] = D_d(t)."""
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
            bitorder="little", count=diagonals,
        )
        state = ((state << 1) ^ (state | (state >> 1))) & mask
    return out


def centre_prefix(n: int = 16) -> str:
    steps = n + 2
    width = 2 * steps + 3
    centre = steps + 1
    mask = (1 << width) - 1
    state = 1 << centre
    bits = []
    for _ in range(n):
        bits.append((state >> centre) & 1)
        state = ((state << 1) ^ (state | (state >> 1))) & mask
    return "".join(str(b) for b in bits)


def check_recursion(dg: np.ndarray) -> dict:
    """Bit-exact check of (R) over every diagonal d >= 2 and every t."""
    bad = []
    for d in range(2, dg.shape[1]):
        lhs = dg[1:, d]
        rhs = dg[:-1, d - 2] ^ (dg[:-1, d - 1] | dg[:-1, d])
        if not np.array_equal(lhs, rhs):
            first = int(np.flatnonzero(lhs != rhs)[0])
            bad.append({"d": d, "first_bad_t": first})
            if len(bad) >= 5:
                break
    return {
        "checked_diagonals": int(dg.shape[1] - 2),
        "checked_steps": int(dg.shape[0] - 1),
        "mismatches": bad,
        "ok": not bad,
    }


def base_cases(dg: np.ndarray) -> dict:
    """D_0 == 1 for t>=0 and D_1 == 1 for t>=1 (D_1(0)=0: the cone has one cell)."""
    return {
        "D0_all_ones_from_t0": bool(dg[:, 0].all()),
        "D1_all_ones_from_t1": bool(dg[1:, 1].all()),
        "D1_at_t0": int(dg[0, 1]),
        "ok": bool(dg[:, 0].all() and dg[1:, 1].all()),
    }


def tail_period(col: np.ndarray, tail: int) -> int | None:
    seg = col[-tail:]
    for p in PERIOD_CANDIDATES:
        if len(seg) > 2 * p and np.array_equal(seg[:-p], seg[p:]):
            return p
    return None


def period_study(dg: np.ndarray, tail: int) -> dict:
    periods = [tail_period(dg[:, d], tail) for d in range(dg.shape[1])]
    hist = Counter(periods)
    finite = [p for p in periods if p is not None]
    over16 = [d for d, p in enumerate(periods) if p is not None and p > 16]
    unsettled = [d for d, p in enumerate(periods) if p is None]
    return {
        "tail_window": tail,
        "histogram": {str(k): v for k, v in sorted(hist.items(), key=lambda kv: (kv[0] is None, kv[0]))},
        "max_finite_period": max(finite) if finite else None,
        "n_settled": len(finite),
        "n_unsettled_in_tail": len(unsettled),
        "unsettled_d_range": [min(unsettled), max(unsettled)] if unsettled else None,
        "settled_with_period_over_16": over16[:20],
        "period_16_conjecture_holds": not over16,
    }


def pattern_map_step(u: int, v: int, period: int = 16) -> tuple[int | None, str]:
    """(w_{d-2}, w_{d-1}) -> w_d on period-bit words.

    phi_t(x) = u[t] XOR (v[t] OR x) is constant when v[t]=1 and the bijection
    x -> u[t] XOR x when v[t]=0. If v is not identically zero over the period the
    one-period composite is constant, so the eventual w_d is unique and does not
    depend on the transient. If v == 0 the composite is affine and the period may
    double -- flagged, not guessed.
    """
    if v == 0:
        return None, "v_is_zero"
    for x0 in (0, 1):
        x = x0
        seq = []
        for t in range(period):
            seq.append(x)
            x = (u >> t & 1) ^ ((v >> t & 1) | x)
        if x == x0:
            w = 0
            for t in range(period):
                if seq[t]:
                    w |= 1 << t
            return w, "ok"
    return None, "period_doubles"


def pattern_map_study(dg: np.ndarray, period: int = 16, tail: int = 4096) -> dict:
    """Check the pattern map against measurement, and look for an orbit cycle."""
    steps, diagonals = dg.shape[0] - 1, dg.shape[1]
    base = max(0, steps - tail)
    pat = np.zeros(diagonals, dtype=np.int64)
    settled = np.zeros(diagonals, dtype=bool)
    for d in range(diagonals):
        col = dg[base:, d]
        if len(col) > 2 * period and np.array_equal(col[:-period], col[period:]):
            settled[d] = True
            w = 0
            for i in range(period):
                if col[i]:
                    w |= 1 << ((base + i) % period)
            pat[d] = w

    agree = mismatch = ambiguous = 0
    for d in range(2, diagonals):
        if settled[d - 2] and settled[d - 1] and settled[d]:
            w, flag = pattern_map_step(int(pat[d - 2]), int(pat[d - 1]), period)
            if flag != "ok":
                ambiguous += 1
            elif w == int(pat[d]):
                agree += 1
            else:
                mismatch += 1

    seen: dict[tuple[int, int], int] = {}
    cycle = None
    n_pairs = 0
    for d in range(diagonals - 1):
        if settled[d] and settled[d + 1]:
            n_pairs += 1
            key = (int(pat[d]), int(pat[d + 1]))
            if key in seen:
                cycle = {"first_d": seen[key], "repeat_d": d, "period_in_d": d - seen[key]}
                break
            seen[key] = d

    return {
        "period": period,
        "n_settled": int(settled.sum()),
        "map_vs_measurement": {"agree": agree, "mismatch": mismatch,
                               "ambiguous_v_zero": ambiguous,
                               "ok": mismatch == 0},
        "n_pairs_examined": n_pairs,
        "n_distinct_pairs": len(seen),
        "orbit_cycle": cycle,
        "n_distinct_patterns": len(set(int(pat[d]) for d in range(diagonals) if settled[d])),
        "finite_orbit_proof_available": cycle is not None,
    }


def generator_check(dg: np.ndarray, period: int = 16, seed_d: int = 256,
                    margin: float = 1.45, tail: int = 4096) -> dict:
    """CERTIFICATE: reproduce the settled wedge from an O(t)-size description.

    Seed w_0..w_{seed_d} from a short simulation (this covers the finitely many
    ambiguous all-zero diagonals), generate every later word with the O(1)
    pattern map, then predict a(t,-t+d) = w_d[t mod period] for every cell with
    t > margin*d + 200 and compare against the actual CA. Zero mismatches means
    the whole settled region is recoverable from the word list alone.
    """
    steps, diagonals = dg.shape[0] - 1, dg.shape[1]
    base = max(0, steps - tail)

    def from_sim(d: int) -> int | None:
        col = dg[base:, d]
        if len(col) <= 2 * period or not np.array_equal(col[:-period], col[period:]):
            return None
        w = 0
        for i in range(period):
            if col[i]:
                w |= 1 << ((base + i) % period)
        return w

    words: list[int | None] = [None] * diagonals
    for d in range(min(seed_d, diagonals)):
        words[d] = from_sim(d)

    fallbacks = 0
    for d in range(seed_d, diagonals):
        u, v = words[d - 2], words[d - 1]
        got = pattern_map_step(u, v, period)[0] if (u is not None and v is not None) else None
        if got is None:
            fallbacks += 1
            got = from_sim(d)
        words[d] = got

    tmod = np.arange(steps + 1) % period
    checked = wrong = 0
    for d in range(seed_d, diagonals):
        if words[d] is None:
            continue
        t0 = int(margin * d) + 200
        if t0 >= steps:
            break
        pred = (words[d] >> tmod[t0:steps + 1]) & 1
        act = dg[t0:steps + 1, d]
        checked += int(act.size)
        wrong += int((pred != act).sum())

    return {
        "period": period,
        "seed_diagonals": seed_d,
        "margin": margin,
        "words_generated_by_map": int(max(0, diagonals - seed_d) - fallbacks),
        "fallbacks_to_simulation": fallbacks,
        "cells_checked": checked,
        "mismatches": wrong,
        "description_bits": int(diagonals * period),
        "compression_ratio": round(checked / (diagonals * period), 1) if diagonals else None,
        "ok": wrong == 0 and checked > 0,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--diagonals", type=int, default=3000)
    ap.add_argument("--tail", type=int, default=4000,
                    help="trailing window used to read off the eventual period")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args(argv)

    if args.diagonals > args.steps:
        ap.error("--diagonals must not exceed --steps")

    t0 = time.perf_counter()
    cp = centre_prefix()
    if cp != OEIS_A051023_PREFIX:
        raise RuntimeError(f"centre prefix {cp!r} != OEIS A051023 {OEIS_A051023_PREFIX!r}")
    dg = left_diagonals(args.steps, args.diagonals)
    rec = check_recursion(dg)
    base = base_cases(dg)
    per = period_study(dg, min(args.tail, args.steps // 2))
    pmap = pattern_map_study(dg, tail=min(args.tail, args.steps // 2))
    gen = generator_check(dg, tail=min(args.tail, args.steps // 2))

    out = {
        "artifact_type": "rule30.diagonal_recursion",
        "artifact_version": ARTIFACT_VERSION,
        "params": {"steps": args.steps, "diagonals": args.diagonals, "tail": args.tail},
        "centre_prefix_A051023": {"prefix": cp, "ok": True},
        "base_cases": base,
        "recursion_check": rec,
        "period_study": per,
        "pattern_map_study": pmap,
        "generator_certificate": gen,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }

    if args.pretty:
        w = sys.stderr
        w.write(f"\ncentre prefix {cp} == OEIS A051023   OK\n")
        w.write(f"base cases D_0==1, D_1==1 (t>=1): {'OK' if base['ok'] else 'FAIL'}\n")
        w.write(f"recursion D_d(t+1) = D_(d-2) XOR (D_(d-1) OR D_d) over "
                f"{rec['checked_diagonals']} diagonals x {rec['checked_steps']} steps: "
                f"{'PASS' if rec['ok'] else 'FAIL'}\n")
        w.write("\ntail-period histogram\n")
        for k, v in per["histogram"].items():
            w.write(f"  period {k:>5}: {v:>7}\n")
        w.write(f"\nmax finite period: {per['max_finite_period']}\n")
        w.write(f"settled diagonals with period > 16: "
                f"{len(per['settled_with_period_over_16'])}\n")
        w.write(f"period-16 conjecture: "
                f"{'HOLDS on this range' if per['period_16_conjecture_holds'] else 'REFUTED'}\n")
        m = pmap["map_vs_measurement"]
        w.write("\npattern map (w_(d-2), w_(d-1)) -> w_d on 16-bit words\n")
        w.write(f"  agree {m['agree']}, mismatch {m['mismatch']}, "
                f"ambiguous(v=0) {m['ambiguous_v_zero']}  -> "
                f"{'PASS' if m['ok'] else 'FAIL'}\n")
        w.write(f"  distinct pairs {pmap['n_distinct_pairs']}/{pmap['n_pairs_examined']}"
                f", distinct patterns {pmap['n_distinct_patterns']}\n")
        w.write("\nO(t) generator certificate\n")
        w.write(f"  words by map {gen['words_generated_by_map']}, "
                f"fallbacks {gen['fallbacks_to_simulation']}\n")
        w.write(f"  cells checked {gen['cells_checked']:,}, "
                f"mismatches {gen['mismatches']}  -> "
                f"{'PASS' if gen['ok'] else 'FAIL'}\n")
        w.write(f"  description {gen['description_bits']} bits, "
                f"compression {gen['compression_ratio']}x\n")
        w.write("  finite-orbit proof: "
                + (f"AVAILABLE {pmap['orbit_cycle']}\n"
                   if pmap["finite_orbit_proof_available"]
                   else "unavailable (orbit does not cycle in this range)\n"))
        w.write(f"\nelapsed {out['elapsed_s']}s\n\n")
        w.flush()

    if args.out:
        from pathlib import Path
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if (rec["ok"] and base["ok"] and gen["ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

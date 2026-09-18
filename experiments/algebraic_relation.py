#!/usr/bin/env python
"""Minimal algebraic-relation budget C*(N) for a binary sequence over F_2.

Christol's theorem: a sequence over F_2 is 2-automatic exactly when its
generating function f(x) = sum a_n x^n is algebraic over F_2(x). So the DFAO
question the repo answers by SAT at n=64 is the same question as

    does there exist a nonzero relation  sum_{i<=D} P_i(x) f(x)^i = 0 ?

with deg P_i <= E. That reformulation buys three things. The search is a kernel
over F_2 rather than SAT, so it runs at N = 10^4 instead of n = 64. The model
class is parameterised by a coefficient count C = (D+1)(E+1), which is a far
larger and more smoothly tunable budget than a DFAO state count. And an
algebraic relation is a *predictor*: it can be tested on bits it never saw.

That last point is what this instrument is built around, because it escapes the
squeeze that makes a pure exclusion uninformative. The repo's counting bound
wants log2|M| >= N or the negative says nothing; the forced-positive gate wants
C <= N or a kernel exists by dimension alone. Those meet at C ~ N, so a single
fit at the threshold proves nothing either way. Extrapolation breaks the tie:

    fit the relation on the first N bits, then check it against bits N..N+H
    that took no part in the fit.

A relation that survives that is a genuine finite shortcut and a Problem-3
object. One that does not was overfitting, whatever its parameter count.

Report C*(N) as a curve against controls, never a point:

    thue-morse  positive control -- 2-automatic, so C*(N) must PLATEAU
    random      null             -- C*(N) must track N
    center      the prize object

A plateau in the center curve would be a shortcut. Growth tracking the null is
the expected outcome and is an observation about a finite prefix, not a proof of
non-automaticity: no finite N rules out a relation with a larger budget.

See docs/theory/README.md sections 2 and 5 for the two vacuity gates.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prize_lab import sequence_bits


def bits_to_poly(bits):
    """Pack a bit list into a Python int, bit k = coefficient of x^k."""
    value = 0
    for i, b in enumerate(bits):
        if b:
            value |= 1 << i
    return value


def clmul(a, b, limit):
    """Carry-less (F_2) product of two packed polynomials, truncated to x^limit.

    Iterating over the sparser operand keeps this at O(popcount * limit/64)
    word operations, which is what makes N = 10^4 cheap in pure Python.
    """
    mask = (1 << limit) - 1
    a &= mask
    b &= mask
    if a.bit_count() > b.bit_count():
        a, b = b, a
    result = 0
    shift = 0
    while b:
        if b & 1:
            result ^= (a << shift) & mask
        b >>= 1
        shift += 1
    return result


def powers(f, degree, limit):
    """f^0 .. f^degree truncated at x^limit."""
    out = [1]
    for _ in range(degree):
        out.append(clmul(out[-1], f, limit))
    return out


def kernel_vector(columns, rows):
    """One nonzero F_2 kernel vector of the matrix whose columns are given.

    Columns are packed as row-bitmaps; elimination tracks the combination that
    produced each pivot, so a dependent column hands back its own relation.
    """
    mask = (1 << rows) - 1
    pivots = {}                      # leading row index -> (residual, combo)
    for index, column in enumerate(columns):
        residual = column & mask
        combo = 1 << index
        while residual:
            lead = residual.bit_length() - 1
            if lead not in pivots:
                pivots[lead] = (residual, combo)
                break
            other, other_combo = pivots[lead]
            residual ^= other
            combo ^= other_combo
        else:
            return combo            # column is dependent: combo annihilates
    return 0


def relation_for(bits, degree, ext_degree, fit_bits):
    """Search for sum_i P_i(x) f^i = 0 on the first `fit_bits` coefficients."""
    f = bits_to_poly(bits)
    fp = powers(f, degree, fit_bits)
    columns, labels = [], []
    for i in range(degree + 1):
        for e in range(ext_degree + 1):
            columns.append((fp[i] << e) & ((1 << fit_bits) - 1))
            labels.append((i, e))
    combo = kernel_vector(columns, fit_bits)
    if not combo:
        return None
    terms = [labels[k] for k in range(len(labels)) if (combo >> k) & 1]
    return terms


def residual_bits(bits, terms, horizon):
    """Coefficients of sum_i P_i(x) f^i out to x^horizon; all zero means it holds."""
    f = bits_to_poly(bits[:horizon])
    degree = max(i for i, _ in terms)
    fp = powers(f, degree, horizon)
    total = 0
    for i, e in terms:
        total ^= (fp[i] << e) & ((1 << horizon) - 1)
    return total


def extrapolates(kind, terms, fit_bits, holdout, seed):
    """Does a relation fitted on `fit_bits` still hold on unseen coefficients?

    This is the whole point of the instrument. A relation at C ~ N is expected
    to fit by dimension alone; only one that predicts coefficients it never saw
    is evidence of structure.
    """
    horizon = fit_bits + holdout
    bits = sequence_bits(kind, horizon, seed=seed)
    residual = residual_bits(bits, terms, horizon)
    # Coefficients below fit_bits were forced to zero by the fit; the holdout
    # window is the only part that carries information.
    unseen = residual >> fit_bits
    return unseen == 0, unseen.bit_count()


def budget_curve(kind, fit_bits, max_degree, max_ext, holdout, seed):
    """Smallest coefficient budget C = (D+1)(E+1) admitting a relation."""
    found = None
    for total in range(2, (max_degree + 1) * (max_ext + 1) + 1):
        for degree in range(1, min(max_degree, total - 1) + 1):
            if total % (degree + 1):
                continue
            ext = total // (degree + 1) - 1
            if ext < 0 or ext > max_ext:
                continue
            terms = relation_for(sequence_bits(kind, fit_bits, seed=seed),
                                 degree, ext, fit_bits)
            if terms is None:
                continue
            holds, misses = extrapolates(kind, terms, fit_bits, holdout, seed)
            found = {"coefficients": total, "degree": degree, "ext_degree": ext,
                     "terms": len(terms), "extrapolates": holds,
                     "holdout_mismatches": misses}
            if holds:
                return found
            # A fit that fails its holdout is overfitting, not a shortcut: keep
            # looking, but remember the smallest budget that fit at all.
    return found


def run(kinds, sizes, max_degree, max_ext, holdout, seed):
    rows = []
    for kind in kinds:
        for fit_bits in sizes:
            started = time.monotonic()
            result = budget_curve(kind, fit_bits, max_degree, max_ext, holdout, seed)
            rows.append({"sequence": kind, "fit_bits": fit_bits,
                         "holdout": holdout, "elapsed_s": round(time.monotonic() - started, 3),
                         **(result or {"coefficients": None, "extrapolates": False})})
            print(f"  {kind:<11} N={fit_bits:<6} "
                  f"C*={rows[-1]['coefficients']} "
                  f"extrapolates={rows[-1]['extrapolates']}", file=sys.stderr, flush=True)
    return rows


def self_test():
    """Thue-Morse must be found algebraic and must extrapolate; random must not.

    Without both halves the instrument is unfalsifiable: a method that never
    finds structure trivially "excludes" everything, and one that always finds
    it is fitting noise.
    """
    ok = True
    tm = budget_curve("thue-morse", 256, 6, 6, 256, 0)
    print(f"  [{'PASS' if tm and tm['extrapolates'] else 'FAIL'}] "
          f"thue-morse admits an extrapolating relation (C*={tm and tm['coefficients']})")
    ok &= bool(tm and tm["extrapolates"])

    tm_big = budget_curve("thue-morse", 1024, 6, 6, 1024, 0)
    plateau = bool(tm and tm_big and tm_big["coefficients"] == tm["coefficients"])
    print(f"  [{'PASS' if plateau else 'FAIL'}] thue-morse C* plateaus "
          f"(N=256 -> {tm and tm['coefficients']}, N=1024 -> {tm_big and tm_big['coefficients']})")
    ok &= plateau

    rnd = budget_curve("random", 256, 6, 6, 256, 30)
    print(f"  [{'PASS' if not (rnd and rnd['extrapolates']) else 'FAIL'}] "
          f"random admits no extrapolating relation in the same budget")
    ok &= not (rnd and rnd["extrapolates"])
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--sequences", default="thue-morse,random,center")
    p.add_argument("--sizes", default="256,512,1024,2048")
    p.add_argument("--max-degree", type=int, default=8)
    p.add_argument("--max-ext-degree", type=int, default=8)
    p.add_argument("--holdout", type=int, default=0,
                   help="unseen coefficients used to test a fit (default: equal to N)")
    p.add_argument("--seed", type=int, default=30)
    p.add_argument("--out", type=Path)
    p.add_argument("--pretty", action="store_true")
    args = p.parse_args()

    if args.self_test:
        print("algebraic_relation self-test")
        ok = self_test()
        print("self-test OK" if ok else "self-test FAILED")
        return 0 if ok else 1

    sizes = [int(s) for s in args.sizes.split(",")]
    kinds = args.sequences.split(",")
    rows = []
    for size in sizes:
        holdout = args.holdout or size
        rows += run(kinds, [size], args.max_degree, args.max_ext_degree, holdout, args.seed)
    report = {"artifact_type": "rule30.algebraic_relation_curve",
              "seed_convention": "single-black-cell for 'center'; controls are not prize objects",
              "max_degree": args.max_degree, "max_ext_degree": args.max_ext_degree,
              "random_seed": args.seed, "results": rows,
              "limits": "Finite-prefix search over bounded-degree algebraic relations. "
                        "A negative excludes only this budget at this N and is not a "
                        "proof of non-automaticity. A relation that extrapolates would "
                        "be a finite shortcut and must be re-checked independently."}
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2) if args.pretty else json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

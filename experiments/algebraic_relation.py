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
    if a.bit_count() < b.bit_count():
        a, b = b, a          # walk the sparser operand, which is now b
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
            candidate = {"coefficients": total, "degree": degree, "ext_degree": ext,
                         "terms": len(terms), "extrapolates": holds,
                         "holdout_mismatches": misses}
            if holds:
                return candidate
            # A fit that fails its holdout is overfitting, not a shortcut: keep
            # looking, but remember the SMALLEST budget that fit at all. Only
            # record the first, since `total` ascends -- overwriting here would
            # report the largest fitting budget and misname it C*.
            if found is None:
                found = candidate
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

    # The budget must exceed N so a fit is forced and the holdout is actually
    # exercised; otherwise this passes vacuously by finding nothing at all.
    rnd = budget_curve("random", 64, 8, 8, 64, 30)
    fitted = bool(rnd)
    print(f"  [{'PASS' if fitted else 'FAIL'}] random fits at a forced budget "
          f"(C*={rnd and rnd['coefficients']} > N=64), so the holdout runs")
    ok &= fitted
    print(f"  [{'PASS' if fitted and not rnd['extrapolates'] else 'FAIL'}] "
          f"that forced fit does NOT extrapolate "
          f"({rnd and rnd['holdout_mismatches']} holdout mismatches)")
    ok &= bool(fitted and not rnd["extrapolates"])
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
    p.add_argument("--complexity", action="store_true",
                   help="report the minimal-budget curve C*(N), which is always "
                        "defined, instead of presence/absence at a fixed budget")
    p.add_argument("--null-seeds", default="30,7,99,3,17,42,77",
                   help="random-null band; the repo's standard is 7 seeds")
    args = p.parse_args()

    if args.complexity:
        sizes = [int(s) for s in args.sizes.split(",")]
        seeds = [int(s) for s in args.null_seeds.split(",")]
        rows = []
        for size in sizes:
            for kind in args.sequences.split(","):
                if kind == "random":
                    for seed in seeds:
                        started = time.monotonic()
                        point = complexity_point(kind, size, args.max_degree, seed)
                        rows.append({"sequence": kind, "seed": seed, "fit_bits": size,
                                     "elapsed_s": round(time.monotonic() - started, 3),
                                     **(point or {"coefficients": None})})
                else:
                    started = time.monotonic()
                    point = complexity_point(kind, size, args.max_degree, 0)
                    rows.append({"sequence": kind, "fit_bits": size,
                                 "elapsed_s": round(time.monotonic() - started, 3),
                                 **(point or {"coefficients": None})})
                r = rows[-1]
                print(f"  {kind:<11} N={size:<6} C*={r['coefficients']} "
                      f"C*/N={r.get('ratio_to_n')}", file=sys.stderr, flush=True)
        band = {}
        for size in sizes:
            nulls = [r["coefficients"] for r in rows
                     if r["sequence"] == "random" and r["fit_bits"] == size
                     and r["coefficients"] is not None]
            centre = next((r["coefficients"] for r in rows
                           if r["sequence"] == "center" and r["fit_bits"] == size), None)
            if nulls:
                band[str(size)] = {"null_min": min(nulls), "null_max": max(nulls),
                                   "null_seeds": len(nulls), "center": centre,
                                   "center_inside_band": centre is not None
                                   and min(nulls) <= centre <= max(nulls)}
        report = {"artifact_type": "rule30.algebraic_complexity_curve",
                  "quantity": "C*(N) = min over D of (D+1)(E_min(D)+1), the smallest "
                              "coefficient budget admitting a relation over F_2(x). "
                              "Always defined: a fit is forced once the column count "
                              "passes N.",
                  "max_degree": args.max_degree, "null_band": band, "results": rows,
                  "limits": "Finite prefix. C*/N near 1 says the algebraic complexity is "
                            "maximal on this prefix, the direct analogue of L(n)=n/2 for "
                            "LFSRs. It is not a proof of non-automaticity."}
        if args.out:
            args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2) if args.pretty else json.dumps(report))
        return 0

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




# --- Algebraic complexity: the curve where C*(N) is defined rather than absent ---
#
# Reporting "no relation at budget C" is an absence, and the ledger rightly grades
# an absence as an observation. The measurable quantity underneath it is the
# budget at which a relation DOES appear, which always exists: once the column
# count passes N a kernel is forced. So instead of asking "is there a fit under
# this budget", ask "what is the smallest budget that fits", and read the answer
# against the null.
#
# For a fixed algebraic degree D the minimal coefficient degree E is
# ordering-independent, which C = (D+1)(E+1) alone is not, so E_min(D) is the
# primitive and C* is the minimum of (D+1)(E_min+1) over D.
#
# The expected readings, and what each would mean:
#   thue-morse   C* constant in N                 -- algebraic, a real shortcut
#   random       C*/N -> 1                        -- maximal, no structure
#   center       C*/N -> 1 would be the direct analogue of L(n)=n/2 for LFSRs,
#                                                   i.e. maximal algebraic
#                                                   complexity, a positive
#                                                   structural statement rather
#                                                   than an absence

def min_ext_for_degree(bits, degree, fit_bits, ext_cap):
    """Smallest E with a relation sum_{i<=degree} P_i(x) f^i = 0 on fit_bits terms.

    Columns are added in groups of constant e, so the first group that closes a
    dependency gives E directly. One elimination pass serves every E.
    """
    mask = (1 << fit_bits) - 1
    fp = powers(bits_to_poly(bits), degree, fit_bits)
    pivots = {}
    for ext in range(ext_cap + 1):
        for i in range(degree + 1):
            residual = (fp[i] << ext) & mask
            while residual:
                lead = residual.bit_length() - 1
                if lead not in pivots:
                    pivots[lead] = residual
                    break
                residual ^= pivots[lead]
            else:
                return ext          # column collapsed: a relation exists at this E
    return None


def complexity_point(kind, fit_bits, max_degree, seed):
    """C*(N) = min over D of (D+1)(E_min(D)+1), with the D that achieves it."""
    bits = sequence_bits(kind, fit_bits, seed=seed)
    best = None
    for degree in range(1, max_degree + 1):
        # Beyond this E the column count exceeds fit_bits and a fit is forced,
        # so there is nothing to learn from searching further.
        ext_cap = fit_bits // (degree + 1) + 1
        ext = min_ext_for_degree(bits, degree, fit_bits, ext_cap)
        if ext is None:
            continue
        coefficients = (degree + 1) * (ext + 1)
        if best is None or coefficients < best["coefficients"]:
            best = {"coefficients": coefficients, "degree": degree, "ext_degree": ext}
    if best:
        best["ratio_to_n"] = round(best["coefficients"] / fit_bits, 4)
    return best


def complexity_curve(kinds, sizes, max_degree, seed):
    rows = []
    for fit_bits in sizes:
        for kind in kinds:
            started = time.monotonic()
            point = complexity_point(kind, fit_bits, max_degree, seed)
            rows.append({"sequence": kind, "fit_bits": fit_bits,
                         "elapsed_s": round(time.monotonic() - started, 3),
                         **(point or {"coefficients": None})})
            r = rows[-1]
            print(f"  {kind:<11} N={fit_bits:<6} C*={r['coefficients']} "
                  f"C*/N={r.get('ratio_to_n')} (D={r.get('degree')},E={r.get('ext_degree')})",
                  file=sys.stderr, flush=True)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())

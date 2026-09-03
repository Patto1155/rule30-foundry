#!/usr/bin/env python
"""Experiment C2 - Algebraic Annihilator Search on the Single-Seed Center Column.

Prize Problem 2 asks for a shortcut to the n-th center-column bit. Experiment S
closed the *linear* form of that question: the center column has maximal linear
complexity, L(n) = n/2, so no LFSR generates it. This experiment asks the next
question up the degree ladder, which the ledger records as not yet asked.

Rule 30's own update is degree 2 in ANF:

    a(t+1, i) = a(t,i-1) XOR a(t,i) XOR a(t,i+1) XOR a(t,i)*a(t,i+1)

so low-degree GF(2) structure is exactly the kind a shortcut might inherit.
Following standard algebraic cryptanalysis (Courtois-Meier), we look for an
*annihilator*: a nonzero polynomial f of degree <= d over the w bits of a
sliding window such that

    f(s[i], s[i+1], ..., s[i+w-1]) = 0   for every position i.

Such an f is a constraint the stream satisfies everywhere - a shortcut
candidate. Its absence, at parameters where absence is not automatic, is a
negative that actually constrains Rule 30.

Method
------
The monomial basis of degree <= d in w variables has

    D = sum_{k=0..d} C(w, k)

elements. Evaluating every monomial at every observed window gives a matrix M
with D columns; f exists iff M has a nonzero right kernel. So the whole
question is the GF(2) rank of M.

Why both directions can be vacuous
----------------------------------
`CLAUDE.md` rule 1 says to run the counting bound before any "searched class M,
found no fit" experiment. This class fails in *both* directions, so
`experiments/counting_bound.py` gained two gates for it (`annihilator_verdict`):

1. **Coverage vacuity - a guaranteed negative.** If the observed windows
   exhaust GF(2)^w, then f vanishing on all of them means f vanishes
   everywhere, so f = 0. "No annihilator" is then a restatement of "the windows
   cover the space". Measured on the golden 10M stream, coverage is complete
   for w <= 18 and 99.99% at w = 20, so any search at w <= 20 is close to
   worthless no matter what it returns. This is the same trap that voided the
   small-DFAO negatives, in a different model class.

2. **Dimension vacuity - a guaranteed positive.** Fewer than D independent rows
   leaves a kernel by dimension alone. We require D + margin distinct windows,
   putting the odds that random data clears the bar at about 2^-margin.

Controls
--------
A negative control that passes while testing nothing is worse than none
(`docs/handover/CURRENT.md`). So the run includes a **positive** control: a
sequence built to satisfy a known relation, which the search must *find*. If
the positive control comes back clean, the machinery is broken and the negative
on Rule 30 means nothing. A random-stream control is also run, and must come
back full rank.

Usage:
    python experiments/algebraic_annihilator.py --pretty
    python experiments/algebraic_annihilator.py --self-test
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import counting_bound  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_TYPE = "rule30.algebraic_annihilator"

# The golden reference is tracked in git, so this experiment runs on a fresh
# clone. center_col_10M.bin is gitignored and identical where present.
DEFAULT_INPUT = REPO_ROOT / "data" / "golden" / "center_col_golden_10M.bin"

MAX_WINDOW = 32  # window codes are packed into uint64 with room to spare


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_bits(path: Path, max_bits: int | None = None) -> np.ndarray:
    """Load a packed center-column dump as a 0/1 uint8 array.

    bitorder='little' is mandatory here: gpu/rule30_sim.py writes LSB-first and
    NumPy defaults to MSB-first. A bare call reverses each 8-bit block, which
    changes 49.95% of positions while leaving every aggregate statistic - the
    bit mean included - identical. See tools/lint_bitorder.py.
    """
    raw = np.fromfile(path, dtype=np.uint8)
    bits = np.unpackbits(raw, bitorder='little')
    if max_bits is not None:
        bits = bits[:max_bits]
    return bits.astype(np.uint8)


def window_codes(bits: np.ndarray, w: int) -> np.ndarray:
    """Every length-w sliding window, packed one per uint64 (bit j = s[i+j])."""
    if not 1 <= w <= MAX_WINDOW:
        raise ValueError(f"window {w} outside 1..{MAX_WINDOW}")
    n = bits.size - w + 1
    if n <= 0:
        raise ValueError(f"stream of {bits.size} bits is shorter than window {w}")
    codes = np.zeros(n, dtype=np.uint64)
    for j in range(w):
        codes |= bits[j:j + n].astype(np.uint64) << np.uint64(j)
    return codes


# --------------------------------------------------------------------------
# monomials and the matrix
# --------------------------------------------------------------------------

def monomials(w: int, d: int) -> list[tuple[int, ...]]:
    """Monomial exponent sets of degree <= d, constant term first."""
    out: list[tuple[int, ...]] = []
    for k in range(d + 1):
        out.extend(itertools.combinations(range(w), k))
    return out


def build_matrix(codes: np.ndarray, monos: list[tuple[int, ...]]) -> np.ndarray:
    """Monomial matrix as bitset rows: (n_rows, ceil(D/64)) uint64.

    Column c of row r is the value of monomial c at window r. Over GF(2) a
    monomial is the AND of its variables.
    """
    n_rows = codes.size
    n_cols = len(monos)
    n_words = (n_cols + 63) // 64
    rows = np.zeros((n_rows, n_words), dtype=np.uint64)
    for c, mono in enumerate(monos):
        if not mono:                       # constant term
            val = np.ones(n_rows, dtype=np.uint64)
        else:
            val = (codes >> np.uint64(mono[0])) & np.uint64(1)
            for j in mono[1:]:
                val &= (codes >> np.uint64(j)) & np.uint64(1)
        rows[:, c >> 6] |= val << np.uint64(c & 63)
    return rows


def evaluate(codes: np.ndarray, monos: list[tuple[int, ...]],
             coeffs: np.ndarray) -> np.ndarray:
    """Evaluate the polynomial with these coefficients at every window."""
    acc = np.zeros(codes.size, dtype=np.uint8)
    for c, mono in enumerate(monos):
        if not coeffs[c]:
            continue
        if not mono:
            val = np.ones(codes.size, dtype=np.uint8)
        else:
            val = ((codes >> np.uint64(mono[0])) & np.uint64(1)).astype(np.uint8)
            for j in mono[1:]:
                val &= ((codes >> np.uint64(j)) & np.uint64(1)).astype(np.uint8)
        acc ^= val
    return acc


# --------------------------------------------------------------------------
# GF(2) linear algebra
# --------------------------------------------------------------------------

def gf2_rref(rows: np.ndarray, n_cols: int) -> list[int]:
    """Reduce `rows` to RREF in place over GF(2); return the pivot columns."""
    n_rows = rows.shape[0]
    pivots: list[int] = []
    r = 0
    for c in range(n_cols):
        if r >= n_rows:
            break
        word, bit = c >> 6, np.uint64(c & 63)
        below = (rows[r:, word] >> bit) & np.uint64(1)
        nz = np.flatnonzero(below)
        if nz.size == 0:
            continue
        p = r + int(nz[0])
        if p != r:
            rows[[r, p]] = rows[[p, r]]
        # Full reduction: clear this column everywhere except the pivot row, so
        # the kernel can be read straight off without back-substitution order
        # bookkeeping.
        col = (rows[:, word] >> bit) & np.uint64(1)
        col[r] = 0
        sel = np.flatnonzero(col)
        if sel.size:
            rows[sel] ^= rows[r]
        pivots.append(c)
        r += 1
    return pivots


def kernel_basis(rows: np.ndarray, pivots: list[int],
                 n_cols: int) -> list[np.ndarray]:
    """Right-kernel basis of a matrix already in RREF.

    For each free column f, the vector with a 1 at f and, at each pivot column
    p_i, the RREF entry (i, f). Empty when the rank is full.
    """
    pivot_set = set(pivots)
    free = [c for c in range(n_cols) if c not in pivot_set]
    out: list[np.ndarray] = []
    for f in free:
        vec = np.zeros(n_cols, dtype=np.uint8)
        vec[f] = 1
        fw, fb = f >> 6, np.uint64(f & 63)
        for i, p in enumerate(pivots):
            vec[p] = np.uint8((rows[i, fw] >> fb) & np.uint64(1))
        out.append(vec)
    return out


# --------------------------------------------------------------------------
# the search
# --------------------------------------------------------------------------

def search(bits: np.ndarray, w: int, d: int, margin_bits: int = 64,
           label: str = "stream", verbose: bool = False, seed: int = 30) -> dict:
    """Full gated search at one (w, d). Returns a JSON-ready record."""
    t0 = time.time()
    codes = window_codes(bits, w)
    uniq = np.unique(codes)
    n_distinct = int(uniq.size)
    covers = bool(n_distinct == (1 << w))

    gate = counting_bound.annihilator_verdict(
        w, d, n_distinct, margin_bits=margin_bits)

    record = {
        "label": label,
        "window_bits": w,
        "degree": d,
        "n_windows": int(codes.size),
        "n_distinct_windows": n_distinct,
        "covers_window_space": covers,
        "gate": gate,
    }

    if not gate["informative"]:
        record["status"] = "skipped_vacuous"
        record["elapsed_s"] = round(time.time() - t0, 3)
        return record

    monos = monomials(w, d)
    dim = len(monos)

    # Rank is at most D, so D + margin distinct windows settle the question.
    # Full rank on a subset implies full rank on the whole stream, so the
    # negative this produces is not weakened by the subsampling. Only a
    # deficiency needs the full stream, and it gets it below.
    # Sample the distinct windows at random. An evenly spaced slice of
    # np.unique's SORTED output is not a neutral subset: consecutive picks share
    # high-order bits, which manufactures rank deficiency that has nothing to do
    # with the sequence. Observed doing exactly that at w=20 before this was
    # fixed. A seeded permutation keeps the run reproducible.
    rng = np.random.default_rng(seed)
    take = min(n_distinct, dim + margin_bits)
    sample = rng.choice(uniq, size=take, replace=False)
    rows = build_matrix(sample, monos)
    pivots = gf2_rref(rows, dim)
    rank = len(pivots)

    # A deficiency on the first sample may still be a sampling accident, so
    # widen the sample before treating it as a candidate. Full rank never needs
    # this: rank D on any subset already proves rank D overall.
    if rank < dim and n_distinct > take:
        take = min(n_distinct, 8 * (dim + margin_bits))
        sample = rng.choice(uniq, size=take, replace=False)
        rows = build_matrix(sample, monos)
        pivots = gf2_rref(rows, dim)
        rank = len(pivots)

    record.update({
        "monomial_dimension": dim,
        "rows_used": int(take),
        "rank": rank,
        "rank_deficiency": dim - rank,
    })

    if rank == dim:
        record["status"] = "no_annihilator"
        record["annihilators"] = []
    else:
        # Rank deficiency on the subset is only a candidate. Confirm against
        # every window in the stream before calling it a relation.
        confirmed = []
        for vec in kernel_basis(rows, pivots, dim):
            resid = evaluate(codes, monos, vec)
            viol = int(resid.sum())
            confirmed.append({
                "support": [list(monos[c]) for c in np.flatnonzero(vec)][:64],
                "n_terms": int(vec.sum()),
                "violations_on_full_stream": viol,
                "holds_everywhere": viol == 0,
            })
        record["annihilators"] = confirmed
        record["status"] = ("annihilator_found"
                            if any(a["holds_everywhere"] for a in confirmed)
                            else "subset_artifact_only")

    record["elapsed_s"] = round(time.time() - t0, 3)
    if verbose:
        print(f"  {label} w={w} d={d} D={dim} rank={rank} "
              f"-> {record['status']}", file=sys.stderr, flush=True)
    return record


# --------------------------------------------------------------------------
# controls
# --------------------------------------------------------------------------

# The planted relation: a degree-2 NFSR of order 20.
#   s[i+20] = s[i] XOR s[i+3] XOR s[i+17] XOR (s[i+1] AND s[i+11])
# Rearranged, the polynomial x20 + x0 + x3 + x17 + x1*x11 annihilates every
# 21-bit window by construction.
CONTROL_ORDER = 20
CONTROL_TAPS = (0, 3, 17)
CONTROL_QUAD = (1, 11)
CONTROL_WINDOW = CONTROL_ORDER + 1


def positive_control_bits(n: int, seed: int = 30) -> np.ndarray:
    """A stream carrying a planted degree-2 relation the search must find.

    A linear feedback would also be detectable, but it would only exercise the
    degree-1 part of the monomial basis - the part Experiment S already
    covers. The quadratic term makes the control test the machinery this
    experiment actually depends on.

    The order is 20 rather than something tiny because both counting-bound
    gates have to clear: the window space must not be covered, and there must
    be more than D + margin distinct windows. Short registers fail the second.
    Measured: 603,990 distinct 21-bit windows over 2M bits, against D = 232.
    """
    rng = np.random.default_rng(seed)
    k = CONTROL_ORDER
    s = np.zeros(n, dtype=np.uint8)
    s[:k] = rng.integers(0, 2, k)
    for i in range(n - k):
        v = np.uint8(0)
        for t in CONTROL_TAPS:
            v ^= s[i + t]
        v ^= s[i + CONTROL_QUAD[0]] & s[i + CONTROL_QUAD[1]]
        s[i + k] = v
    return s


def random_control_bits(n: int, seed: int = 30) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, n, dtype=np.uint8)


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def self_test() -> int:
    """Prove the search finds a planted relation and rejects a random stream."""
    ok = True

    bits = positive_control_bits(300_000)
    rec = search(bits, w=CONTROL_WINDOW, d=2, margin_bits=64,
                 label="positive-control")
    found = rec.get("status") == "annihilator_found"
    print(f"positive control (planted degree-2 NFSR): {rec.get('status')} "
          f"rank_deficiency={rec.get('rank_deficiency')}", file=sys.stderr)
    if not found:
        print("  FAIL: the search missed a relation that holds by construction",
              file=sys.stderr)
        ok = False

    rnd = random_control_bits(2_000_000)
    rec = search(rnd, w=22, d=2, margin_bits=64, label="random-control")
    if rec.get("status") != "no_annihilator":
        print(f"  FAIL: random stream reported {rec.get('status')}",
              file=sys.stderr)
        ok = False
    else:
        print("random control: no_annihilator (as required)", file=sys.stderr)

    # The coverage gate must refuse a search that cannot say anything.
    rec = search(rnd, w=12, d=2, margin_bits=64, label="coverage-gate")
    if rec.get("status") != "skipped_vacuous":
        print(f"  FAIL: w=12 should be vacuous, got {rec.get('status')}",
              file=sys.stderr)
        ok = False
    else:
        print("coverage gate: w=12 refused as vacuous (as required)",
              file=sys.stderr)

    print("SELF-TEST " + ("PASS" if ok else "FAIL"), file=sys.stderr)
    return 0 if ok else 1


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--max-bits", type=int, default=None)
    ap.add_argument("--windows", type=str, default="22,24,26,28,32",
                    help="comma-separated window sizes w")
    ap.add_argument("--degrees", type=str, default="2,3",
                    help="comma-separated degrees d")
    ap.add_argument("--margin-bits", type=int, default=64)
    ap.add_argument("--controls", action="store_true",
                    help="also run the positive and random controls")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--pretty", action="store_true",
                    help="human table on stderr")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the JSON artifact here as well as stdout")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 2

    bits = load_bits(args.input, args.max_bits)
    windows = [int(x) for x in args.windows.split(",") if x]
    degrees = [int(x) for x in args.degrees.split(",") if x]

    results = []
    for w in windows:
        for d in degrees:
            results.append(search(bits, w, d, args.margin_bits,
                                  label="rule30-center", verbose=args.pretty))

    controls = []
    if args.controls:
        pc = positive_control_bits(300_000)
        controls.append(search(pc, CONTROL_WINDOW, 2, args.margin_bits,
                               label="positive-control", verbose=args.pretty))
        rc = random_control_bits(bits.size)
        for w in windows:
            controls.append(search(rc, w, degrees[0], args.margin_bits,
                                   label="random-control", verbose=args.pretty))

    artifact = {
        "artifact_type": ARTIFACT_TYPE,
        "input": str(args.input.relative_to(REPO_ROOT)),
        "n_bits": int(bits.size),
        "bit_mean": round(float(bits.mean()), 6),
        "margin_bits": args.margin_bits,
        "results": results,
        "controls": controls,
    }

    if args.pretty:
        print(f"\n{'w':>3} {'d':>2} {'D':>8} {'distinct':>10} {'rank':>8} "
              f"{'status':>22}  gate", file=sys.stderr)
        print("-" * 92, file=sys.stderr)
        for r in results:
            print(f"{r['window_bits']:>3} {r['degree']:>2} "
                  f"{r.get('monomial_dimension', 0):>8} "
                  f"{r['n_distinct_windows']:>10} {r.get('rank', 0):>8} "
                  f"{r['status']:>22}  "
                  f"{'informative' if r['gate']['informative'] else 'VACUOUS'}",
                  file=sys.stderr)

    text = json.dumps(artifact, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

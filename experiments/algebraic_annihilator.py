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
    python experiments/algebraic_annihilator.py --self-test     # controls only
    python experiments/algebraic_annihilator.py --verify ART    # the claim
    python experiments/algebraic_annihilator.py --space-time 8
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

# A window code is one uint64, so 64 bits is the hard cap, not a chosen one.
# The Reed-Muller ceiling grows as 2^w while D grows polynomially, so width is
# gated far more loosely than degree: w=64 at d=2 needs only D=2081 columns,
# cheaper than the d=3 grid. Width is the cheap axis; use it before degree.
MAX_WINDOW = 64


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

# A reversed 8-bit block disagrees at ~50% of positions, so a few hundred bits
# make a wrong decode impossible to miss.
VERIFY_BITS = 512

# tools/gen_golden_reference.py packs MSB-first, by documented exception.
GOLDEN_BITORDER = "big"


def naive_center_column(n: int) -> np.ndarray:
    """The center column straight from the rule, no packing involved.

    Deliberately independent of every packed path in this repo: it is the
    reference that decides whether a loaded file really is the center column.
    """
    width = 2 * n + 5
    row = np.zeros(width, dtype=np.uint8)
    row[width // 2] = 1
    out = np.empty(n, dtype=np.uint8)
    for t in range(n):
        out[t] = row[width // 2]
        row = np.roll(row, 1) ^ (row | np.roll(row, -1))
    return out


def load_bits(path: Path,
              max_bits: int | None = None) -> tuple[np.ndarray, str]:
    """Load a packed center-column dump, *verifying* it really is one.

    This repo has two packing conventions and both are correct:
    `gpu/rule30_sim.py` writes LSB-first, and `tools/gen_golden_reference.py`
    writes MSB-first by deliberate documented exception (AGENTS.md - its
    independence from the kernel is the whole point, do not "fix" it). Picking
    the wrong one reverses every 8-bit block: 49.95% of positions differ while
    the bit mean is *identical*, so no aggregate check catches it.

    This experiment shipped exactly that bug on its first run. It read the
    golden file as little-endian, analysed a byte-block-reversed stream for the
    whole grid, and reported a perfectly healthy bit mean of 0.500222 while
    doing it. Caught in review, not by any check here.

    So this does not guess, and does not trust a caller's flag either. It
    decodes both ways and keeps whichever reproduces `naive_center_column`. A
    file matching neither is rejected rather than silently analysed. Returns
    the bits and the confirmed convention, so the artifact can record it.
    """
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size * 8 < VERIFY_BITS:
        raise ValueError(f"{path} holds fewer than {VERIFY_BITS} bits")

    reference = naive_center_column(VERIFY_BITS)
    head = raw[:(VERIFY_BITS + 7) // 8]
    confirmed = None
    for order in ("little", "big"):
        probe = np.unpackbits(head, bitorder=order)[:VERIFY_BITS]
        if np.array_equal(probe, reference):
            confirmed = order
            break

    if confirmed is None:
        raise ValueError(
            f"{path} does not decode to the Rule 30 center column under "
            "either bit order: it is not the single-seed center column, or it "
            "is corrupt. Refusing to analyse it.")

    bits = np.unpackbits(raw, bitorder=confirmed).astype(np.uint8)

    # The prefix fixes the bit order; it does not certify the rest of the file.
    # A dump whose first 512 bits are right but which diverges later -- a tape
    # too narrow, so the light cone reaches the edge and wraps -- would sail
    # through and be recorded as "verified". So a non-golden input is also
    # checked against the golden reference across their whole overlap, which is
    # what tools/verify_data.py does.
    checked_to = VERIFY_BITS
    if path.resolve() != DEFAULT_INPUT.resolve() and DEFAULT_INPUT.exists():
        golden_raw = np.fromfile(DEFAULT_INPUT, dtype=np.uint8)
        golden = np.unpackbits(golden_raw, bitorder=GOLDEN_BITORDER)
        overlap = min(golden.size, bits.size)
        diff = np.flatnonzero(golden[:overlap] != bits[:overlap])
        if diff.size:
            raise ValueError(
                f"{path} diverges from {DEFAULT_INPUT.name} at bit "
                f"{int(diff[0])} of {overlap} overlapping bits "
                f"({diff.size} differ). Decoded under bitorder={confirmed!r}. "
                "Refusing to analyse it.")
        checked_to = overlap

    if max_bits is not None:
        bits = bits[:max_bits]
    if bits.size > checked_to:
        # Say so rather than implying the whole stream was certified.
        print(f"note: {path.name} verified over its first {checked_to:,} bits; "
              f"{bits.size - checked_to:,} beyond that are unchecked",
              file=sys.stderr)
    return bits, confirmed


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

def confirm_on_full_stream(codes: np.ndarray, monos: list[tuple[int, ...]],
                           basis: list[np.ndarray]) -> list[dict]:
    """Return the annihilators holding at EVERY window, not just the sample.

    `basis` spans the kernel of the sampled matrix. A true annihilator vanishes
    on the sample too, so it lies in that span -- but as a combination
    `sum c_j b_j`, and each `b_j` alone may violate somewhere. Checking the
    basis element-wise therefore misses relations this experiment exists to
    find.

    So change coordinates: residual column j is `b_j` evaluated at every
    window, and a combination annihilates the stream exactly when `c` lies in
    the kernel of that residual matrix. Duplicate residual rows carry no extra
    constraint, which collapses 10^7 windows to at most 2^k of them.
    """
    if not basis:
        return []
    k = len(basis)
    residual = np.stack([evaluate(codes, monos, b) for b in basis], axis=1)

    packed = np.zeros((residual.shape[0], (k + 63) // 64), dtype=np.uint64)
    for j in range(k):
        packed[:, j >> 6] |= residual[:, j].astype(np.uint64) << np.uint64(j & 63)
    rows = np.unique(packed, axis=0)

    pivots = gf2_rref(rows, k)
    out = []
    for combo in kernel_basis(rows, pivots, k):
        vec = np.zeros(len(monos), dtype=np.uint8)
        for j in np.flatnonzero(combo):
            vec ^= basis[j]
        if not vec.any():
            continue
        violations = int(evaluate(codes, monos, vec).sum())
        if violations:
            continue
        out.append({
            "support": [list(monos[c]) for c in np.flatnonzero(vec)][:64],
            "n_terms": int(vec.sum()),
            "n_basis_vectors_combined": int(combo.sum()),
            "violations_on_full_stream": 0,
            "holds_everywhere": True,
        })
    return out


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
        # Rank deficiency on the subset is only a candidate, and a true
        # annihilator need not be one of the basis vectors -- it can be a
        # combination whose individual parts each violate somewhere. Testing
        # the basis element-wise would miss it.
        record["annihilators"] = confirm_on_full_stream(
            codes, monos, kernel_basis(rows, pivots, dim))
        record["status"] = ("annihilator_found"
                            if record["annihilators"]
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
# space-time: a closed route, demonstrated rather than asserted
# --------------------------------------------------------------------------

def poly_mul(a: set, b: set) -> set:
    """Multiply two GF(2) polynomials in the Boolean ring (x^2 = x).

    A polynomial is a set of monomials with coefficient 1; a monomial is a
    frozenset of variable indices. Multiplying monomials unions their variable
    sets, because x*x = x, and equal terms cancel over GF(2).
    """
    out: set = set()
    for ma in a:
        for mb in b:
            out ^= {ma | mb}
    return out


def vec_to_poly(vec: np.ndarray, monos: list[tuple[int, ...]]) -> set:
    return {frozenset(monos[c]) for c in np.flatnonzero(vec)}


def poly_to_vec(poly: set, monos: list[tuple[int, ...]]) -> np.ndarray | None:
    """Coefficient vector, or None if any term is outside the basis."""
    index = {frozenset(m): i for i, m in enumerate(monos)}
    vec = np.zeros(len(monos), dtype=np.uint8)
    for m in poly:
        if m not in index:
            return None
        vec[index[m]] ^= 1
    return vec


def ideal_span_within_basis(generators: list[np.ndarray],
                            monos: list[tuple[int, ...]],
                            n_vars: int) -> list[np.ndarray]:
    """The part of the ideal generated by `generators` that fits in `monos`.

    Only the degree-bounded slice matters, because the kernel being classified
    lives in that slice. Multiplying a degree-2 generator by a variable can
    stay degree 2 in the Boolean ring -- x*(x + xy) = x + xy -- so single-
    variable and pairwise products are enumerated and the ones that fit are
    kept. Anything landing outside the basis cannot appear in the kernel and
    is dropped.
    """
    polys = [vec_to_poly(g, monos) for g in generators]
    singles = [{frozenset()}] + [{frozenset([j])} for j in range(n_vars)]

    out: list[np.ndarray] = []
    seen: set = set()

    def keep(poly: set) -> None:
        vec = poly_to_vec(poly, monos)
        if vec is None or not vec.any():
            return
        key = vec.tobytes()
        if key not in seen:
            seen.add(key)
            out.append(vec)

    for g in polys:
        for m in singles:
            keep(poly_mul(g, m))
    for i, gi in enumerate(polys):
        for gj in polys[i + 1:]:
            keep(poly_mul(gi, gj))
    return out


def span_rank(vectors: list[np.ndarray], n_cols: int) -> int:
    if not vectors:
        return 0
    rows = np.zeros((len(vectors), (n_cols + 63) // 64), dtype=np.uint64)
    for r, v in enumerate(vectors):
        for c in np.flatnonzero(v):
            rows[r, c >> 6] |= np.uint64(1) << np.uint64(c & 63)
    return len(gf2_rref(rows, n_cols))


def rule30_field(steps: int, width: int) -> np.ndarray:
    """Rule 30 space-time from the single black cell, as a (steps, width) array."""
    row = np.zeros(width, dtype=np.uint8)
    row[width // 2] = 1
    field = np.empty((steps, width), dtype=np.uint8)
    for t in range(steps):
        field[t] = row
        row = np.roll(row, 1) ^ (row | np.roll(row, -1))
    return field


def spacetime_codes(field: np.ndarray, k: int) -> np.ndarray:
    """Every 2-row x k-column patch, packed into a 2k-bit code.

    Bits 0..k-1 are row t, columns i..i+k-1; bits k..2k-1 are row t+1 over the
    same columns. All-zero patches (the quiescent region outside the light cone)
    are dropped: they are not part of the object and would dominate the sample.
    """
    if 2 * k > MAX_WINDOW:
        raise ValueError(f"2*{k} exceeds the {MAX_WINDOW}-bit code")
    top, bot = field[:-1, :], field[1:, :]
    n_i = field.shape[1] - k + 1
    codes = np.zeros((top.shape[0], n_i), dtype=np.uint64)
    for j in range(k):
        codes |= top[:, j:j + n_i].astype(np.uint64) << np.uint64(j)
        codes |= bot[:, j:j + n_i].astype(np.uint64) << np.uint64(k + j)
    flat = codes.ravel()
    return flat[flat != 0]


def spacetime_demo(k: int = 8, steps: int = 3000, width: int = 7000,
                   margin_bits: int = 64) -> dict:
    """Show that a space-time annihilator search only recovers the local rule.

    Rule 30's own update is degree 2 over a space-time patch:

        a(t+1,i) = a(t,i-1) XOR a(t,i) XOR a(t,i+1) XOR a(t,i)*a(t,i+1)

    so an annihilator search over 2-row patches is guaranteed to succeed, and
    what it finds is the rule and its ideal multiples. That is a forced
    positive of a kind neither counting-bound gate catches: the vacuity is not
    dimensional, it is that the answer is fixed in advance by the definition of
    the object.

    This function is a demonstration, not a search. It exists so the route is
    closed with evidence in the repo rather than by an argument in a PR
    comment, and so nobody re-opens it.
    """
    field = rule30_field(steps, width)
    codes = spacetime_codes(field, k)
    w = 2 * k
    uniq = np.unique(codes)
    n_distinct = int(uniq.size)
    gate = counting_bound.annihilator_verdict(w, 2, n_distinct,
                                              margin_bits=margin_bits)

    out = {
        "patch": f"2x{k}",
        "window_bits": w,
        "n_patches": int(codes.size),
        "n_distinct_patches": n_distinct,
        "gate": gate,
    }
    if not gate["informative"]:
        out["status"] = "skipped_vacuous"
        return out

    monos = monomials(w, 2)
    dim = len(monos)
    # Every distinct patch, not a sample: the point is to classify the kernel
    # exactly, and a sampled kernel can be larger than the true one.
    rows = build_matrix(uniq, monos)
    pivots = gf2_rref(rows, dim)
    rank = len(pivots)

    # The rule in this patch's variable numbering: for offset i,
    # a(t,i) + a(t,i+1) + a(t,i+2) + a(t+1,i+1) + a(t,i+1)*a(t,i+2).
    index = {m: c for c, m in enumerate(monos)}
    rule_vectors = []
    for i in range(k - 2):
        vec = np.zeros(dim, dtype=np.uint8)
        for m in ((i,), (i + 1,), (i + 2,), (k + i + 1,), (i + 1, i + 2)):
            vec[index[m]] ^= 1
        rule_vectors.append(vec)

    kernel = kernel_basis(rows, pivots, dim)

    # Each rule instance must vanish on every patch -- it holds by
    # construction, so a violation would mean the machinery is broken.
    holding = sum(1 for v in rule_vectors
                  if int(evaluate(codes, monos, v).sum()) == 0)

    # How much of the kernel the rule demonstrably accounts for. This is a
    # LOWER BOUND on the ideal's degree-2 slice, not the slice: it enumerates
    # products of the generators with 1, with single variables, and pairwise.
    # An ideal element can also be a sum `sum m_i * r_i` whose terms each
    # exceed degree 2 while the sum collapses back into it, and those are not
    # enumerated. Closing that gap is ideal membership -- a Groebner basis
    # problem -- which is precisely the work reopening this route would need.
    ideal = ideal_span_within_basis(rule_vectors, monos, w)
    ideal_rank_lower_bound = span_rank(ideal, dim)
    unattributed = len(kernel) - ideal_rank_lower_bound

    out.update({
        "monomial_dimension": dim,
        "rank": rank,
        "rank_deficiency": dim - rank,
        "kernel_dimension": len(kernel),
        "rule_instances": len(rule_vectors),
        "rule_instances_holding": holding,
        "ideal_slice_rank_lower_bound": ideal_rank_lower_bound,
        "kernel_dimensions_unattributed": unattributed,
        "status": ("forced_positive_rule_in_kernel"
                   if holding == len(rule_vectors) else "rule_not_recovered"),
        "reading": (
            f"All {holding} instances of the local rule lie in the kernel, so a "
            "space-time annihilator search succeeds by construction: a forced "
            "positive that says nothing about Rule 30. That is what closes the "
            "route. It is NOT established that the kernel is only the rule "
            f"ideal -- {unattributed} of {len(kernel)} dimensions are "
            "unattributed here. They are not evidence of new structure either: "
            "the enumerated ideal slice is a lower bound, and the realisable "
            "patches are themselves restricted (a 2xk patch is fixed by its top "
            "row plus the two edge bits the rule cannot reach, so at most "
            "2^(k+2) of 2^(2k) occur). Deciding them is ideal membership, not a "
            "rank computation."),
    })
    return out


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


def verify_artifact(path: Path, max_dim: int | None = None) -> int:
    """Recompute every claimed row and compare it against the artifact.

    `--self-test` exercises the controls: it proves the machinery finds a
    planted relation and rejects a random stream. It does not load the golden
    input, recompute any claimed rank, or look at the committed artifact -- so
    a stale artifact, or a rank bug confined to the large d=3/d=4 cells, would
    pass it. A Certificate's verifier has to check the certificate, so this
    does, row by row.

    Recomputing the full grid costs about an hour, most of it in w=32 d=4 and
    w=64 d=3. `max_dim` skips rows above a monomial count, for a fast partial
    check -- and the summary says plainly that it was partial.
    """
    artifact = json.loads(path.read_text())
    src = REPO_ROOT / artifact["input"]
    bits, bitorder = load_bits(src)

    if bitorder != artifact.get("bitorder"):
        print(f"FAIL  bitorder: artifact says {artifact.get('bitorder')!r}, "
              f"file decodes as {bitorder!r}", file=sys.stderr)
        return 1
    if int(bits.size) != int(artifact["n_bits"]):
        print(f"FAIL  n_bits: artifact {artifact['n_bits']}, file {bits.size}",
              file=sys.stderr)
        return 1

    checked = skipped = failed = 0
    for row in artifact["results"]:
        w, d = row["window_bits"], row["degree"]
        if max_dim is not None and row.get("monomial_dimension", 0) > max_dim:
            print(f"SKIP  w={w:>2} d={d}  D={row.get('monomial_dimension')} "
                  f"> --verify-max-dim", file=sys.stderr)
            skipped += 1
            continue

        got = search(bits, w, d, artifact["margin_bits"])
        for field in ("status", "n_distinct_windows", "rank",
                      "monomial_dimension"):
            if field in row and row[field] != got.get(field):
                print(f"FAIL  w={w:>2} d={d}  {field}: artifact {row[field]!r}, "
                      f"recomputed {got.get(field)!r}", file=sys.stderr)
                failed += 1
                break
        else:
            print(f"OK    w={w:>2} d={d}  rank={got.get('rank')} "
                  f"{got['status']}", file=sys.stderr)
            checked += 1

    verdict = "FAIL" if failed else ("PARTIAL" if skipped else "OK")
    print(f"\nverify: {checked} rows reproduced, {skipped} skipped, "
          f"{failed} failed  {verdict}", file=sys.stderr)
    if skipped:
        print("  Skipped rows were NOT verified. The certificate covers every "
              "row; a partial run does not.", file=sys.stderr)
    return 1 if failed else 0


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
    ap.add_argument("--verify", type=Path, metavar="ARTIFACT", default=None,
                    help="recompute every claimed row of ARTIFACT and compare. "
                         "This is the certificate verifier; --self-test only "
                         "exercises the controls.")
    ap.add_argument("--verify-max-dim", type=int, default=None,
                    help="with --verify, skip rows whose monomial count "
                         "exceeds this, for a fast partial check")
    ap.add_argument("--space-time", type=int, metavar="K", default=None,
                    help="demonstrate that a 2xK space-time patch search only "
                         "recovers the local rule, then exit")
    ap.add_argument("--pretty", action="store_true",
                    help="human table on stderr")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the JSON artifact here as well as stdout")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if args.verify is not None:
        return verify_artifact(args.verify, args.verify_max_dim)

    if args.space_time is not None:
        rec = spacetime_demo(args.space_time)
        if args.pretty:
            print(f"2x{args.space_time} patch: rank {rec.get('rank')} of "
                  f"{rec.get('monomial_dimension')}, kernel dim "
                  f"{rec.get('kernel_dimension')}, ideal slice >= "
                  f"{rec.get('ideal_slice_rank_lower_bound')}, "
                  f"{rec.get('rule_instances_holding')}/"
                  f"{rec.get('rule_instances')} rule instances hold, "
                  f"{rec.get('kernel_dimensions_unattributed')} unattributed "
                  f"-> {rec.get('status')}", file=sys.stderr)
        print(json.dumps(rec, indent=2))
        return 0

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 2

    bits, bitorder = load_bits(args.input, args.max_bits)
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
        "bitorder": bitorder,
        "bitorder_verified_against": "naive_center_column",
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
        # data/** is excluded from git normalisation and hashed byte-for-byte
        # in data/MANIFEST.sha256, so newline translation would make the
        # artifact platform-dependent and invalidate the manifest. AGENTS.md.
        args.out.write_text(text + "\n", newline="")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

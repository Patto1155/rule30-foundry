#!/usr/bin/env python3
"""Search proof-facing invariants of Rule 30's alternating-trace fibre.

For support radius ``w``, an eternally alternating trace would require every
forced initial bit left of ``-w`` to be zero.  Define ``E(w, phase)`` as the
latest, over all right halves supported in ``1..w``, that the *first* forced
one strictly left of ``-w`` can occur.  A finite value for every ``w`` excludes
an alternating centre for finite configurations; a uniform proved bound on
``E`` would therefore settle eventual period two.

This script computes bounded values of E and tests three deliberately small
induction languages:

* linear horizon bounds ``E(w) <= a*w+b``;
* common phase-aware affine recurrences in spatial depth for the forced left
  bits;
* common phase-aware quadratic recurrences matching the local rule's degree.

Training and holdout radii are separated.  Surviving either finite test is a
proof candidate only; failure is a genuine counterexample to that candidate.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.periodic_trace_fiber import forced_left_prefix, verify_trace


def first_escape_depth(left: np.ndarray, support: int) -> np.ndarray:
    """First forced one at depth > support, or depth+1 when not yet observed."""
    tail = left[:, support:]
    if tail.shape[1] == 0:
        return np.full(left.shape[0], left.shape[1] + 1, dtype=np.int32)
    has_one = tail.any(axis=1)
    return np.where(has_one, tail.argmax(axis=1) + support + 1,
                    left.shape[1] + 1).astype(np.int32)


def horizon_row(support: int, depth: int, phase: int) -> tuple[dict, np.ndarray]:
    rows, left, target, values = forced_left_prefix(
        depth, support, "01", rotation=phase)
    mismatches = verify_trace(rows, target)
    if mismatches:
        raise RuntimeError(f"constructed fibre failed verification: {mismatches}")
    escape = first_escape_depth(left, support)
    saturated = escape > depth
    maximum = int(escape.max())
    return ({
        "support": support,
        "phase": phase,
        "right_halves": len(values),
        "maximum_first_escape_depth": maximum,
        "extremizer_count": int(np.count_nonzero(escape == maximum)),
        "saturated_count": int(saturated.sum()),
    }, left)


def linear_bound_candidates(rows: list[dict], train_max: int) -> list[dict]:
    out = []
    for slope in (1, 2, 3, 4):
        for intercept in range(17):
            train_bad = [r for r in rows if r["support"] <= train_max and
                         r["maximum_first_escape_depth"] > slope * r["support"] + intercept]
            if train_bad:
                continue
            holdout_bad = [r for r in rows if r["support"] > train_max and
                           r["maximum_first_escape_depth"] > slope * r["support"] + intercept]
            out.append({
                "slope": slope,
                "intercept": intercept,
                "train_survives": True,
                "holdout_survives": not holdout_bad,
                "first_holdout_counterexample": holdout_bad[0] if holdout_bad else None,
            })
            break  # Tightest intercept at this slope.
    return out


def _solve_gf2(equations: list[int], variables: int) -> int | None:
    """Solve packed GF(2) equations; bit ``variables`` is the right-hand side."""
    basis: dict[int, int] = {}
    coeff_mask = (1 << variables) - 1
    for original in equations:
        row = original
        while row & coeff_mask:
            pivot = (row & coeff_mask).bit_length() - 1
            if pivot in basis:
                row ^= basis[pivot]
            else:
                basis[pivot] = row
                break
        else:
            if (row >> variables) & 1:
                return None

    solution = 0
    # Each row is pivoted on its highest coefficient, so it can contain only
    # lower unknowns.  Solve those lower pivots first.
    for pivot in sorted(basis):
        row = basis[pivot]
        known_parity = ((row & coeff_mask & solution).bit_count() & 1)
        rhs = (row >> variables) & 1
        if known_parity != rhs:
            solution |= 1 << pivot
    return solution


def fit_affine_recurrence(left: np.ndarray, support: int, memory: int,
                          modulus: int, end_depth: int) -> int | None:
    variables = memory + modulus
    equations: list[int] = []
    start = max(support + 1, memory + 1)
    for depth in range(start, end_depth + 1):
        phase_bit = memory + (depth % modulus)
        histories = left[:, depth - memory - 1:depth - 1]
        targets = left[:, depth - 1]
        for history, target in zip(histories, targets):
            packed = 1 << phase_bit
            # Coefficient j refers to lag j+1; reverse the displayed history.
            for lag, bit in enumerate(history[::-1]):
                if bit:
                    packed |= 1 << lag
            packed |= int(target) << variables
            equations.append(packed)
    return _solve_gf2(equations, variables)


def check_affine_recurrence(left: np.ndarray, support: int, memory: int,
                            modulus: int, solution: int, start_depth: int) -> dict:
    tested = violations = 0
    start = max(support + 1, memory + 1, start_depth)
    lag_mask = (1 << memory) - 1
    for depth in range(start, left.shape[1] + 1):
        intercept = (solution >> (memory + depth % modulus)) & 1
        histories = left[:, depth - memory - 1:depth - 1]
        for history, target in zip(histories, left[:, depth - 1]):
            packed = 0
            for lag, bit in enumerate(history[::-1]):
                packed |= int(bit) << lag
            prediction = intercept ^ ((solution & lag_mask & packed).bit_count() & 1)
            violations += prediction != int(target)
            tested += 1
    return {"tested": tested, "violations": int(violations)}


def _monomial_terms(memory: int, degree: int) -> list[tuple[int, ...]]:
    return [term for d in range(1, degree + 1)
            for term in itertools.combinations(range(memory), d)]


def _history_features(history: np.ndarray,
                      terms: list[tuple[int, ...]]) -> int:
    packed = 0
    for index, term in enumerate(terms):
        value = 1
        for lag in term:
            value &= int(history[-1 - lag])
        packed |= value << index
    return packed


def fit_polynomial_recurrence(left: np.ndarray, support: int, memory: int,
                              degree: int, modulus: int,
                              end_depth: int) -> tuple[int | None, list[tuple[int, ...]]]:
    terms = _monomial_terms(memory, degree)
    variables = len(terms) + modulus
    equations: list[int] = []
    start = max(support + 1, memory + 1)
    for depth in range(start, end_depth + 1):
        phase_bit = len(terms) + (depth % modulus)
        histories = left[:, depth - memory - 1:depth - 1]
        for history, target in zip(histories, left[:, depth - 1]):
            equations.append(_history_features(history, terms) |
                             (1 << phase_bit) | (int(target) << variables))
    return _solve_gf2(equations, variables), terms


def check_polynomial_recurrence(left: np.ndarray, support: int, memory: int,
                                modulus: int, solution: int,
                                terms: list[tuple[int, ...]],
                                start_depth: int) -> dict:
    tested = violations = 0
    term_mask = (1 << len(terms)) - 1
    start = max(support + 1, memory + 1, start_depth)
    for depth in range(start, left.shape[1] + 1):
        intercept = (solution >> (len(terms) + depth % modulus)) & 1
        histories = left[:, depth - memory - 1:depth - 1]
        for history, target in zip(histories, left[:, depth - 1]):
            features = _history_features(history, terms)
            prediction = intercept ^ (
                (solution & term_mask & features).bit_count() & 1)
            violations += prediction != int(target)
            tested += 1
    return {"tested": tested, "violations": int(violations)}


def affine_candidates(train_left: dict[int, np.ndarray], train_support: int,
                      train_depth: int, holdout_left: dict[int, np.ndarray],
                      holdout_support: int) -> list[dict]:
    candidates = []
    for phase in (0, 1):
        for modulus in (1, 2, 4, 8):
            for memory in range(1, 25):
                solution = fit_affine_recurrence(
                    train_left[phase], train_support, memory, modulus, train_depth)
                if solution is None:
                    continue
                validation = check_affine_recurrence(
                    holdout_left[phase], holdout_support, memory, modulus,
                    solution, train_depth + 1)
                candidates.append({
                    "phase": phase,
                    "memory": memory,
                    "depth_modulus": modulus,
                    "coefficient_mask": solution & ((1 << memory) - 1),
                    "phase_intercepts": [
                        (solution >> (memory + j)) & 1 for j in range(modulus)],
                    "holdout": validation,
                })
    return candidates


def quadratic_candidates(train_left: dict[int, np.ndarray], train_support: int,
                         train_depth: int, holdout_left: dict[int, np.ndarray],
                         holdout_support: int) -> list[dict]:
    candidates = []
    for phase in (0, 1):
        for modulus in (1, 2, 4):
            for memory in range(2, 13):
                solution, terms = fit_polynomial_recurrence(
                    train_left[phase], train_support, memory, 2, modulus,
                    train_depth)
                if solution is None:
                    continue
                validation = check_polynomial_recurrence(
                    holdout_left[phase], holdout_support, memory, modulus,
                    solution, terms, train_depth + 1)
                candidates.append({
                    "phase": phase,
                    "memory": memory,
                    "degree": 2,
                    "depth_modulus": modulus,
                    "monomial_count": len(terms),
                    "holdout": validation,
                })
    return candidates


def run(train_max: int, holdout_max: int, depth: int) -> dict:
    if not 1 <= train_max < holdout_max <= 18:
        raise ValueError("require 1 <= train-max < holdout-max <= 18")
    if depth < 3 * holdout_max:
        raise ValueError("depth must be at least three times holdout-max")

    rows: list[dict] = []
    saved: dict[tuple[int, int], np.ndarray] = {}
    for support in range(1, holdout_max + 1):
        for phase in (0, 1):
            entry, left = horizon_row(support, depth, phase)
            rows.append(entry)
            if support in (min(train_max, 8), train_max,
                           min(holdout_max, 12), holdout_max):
                saved[support, phase] = left

    if any(r["saturated_count"] for r in rows):
        raise RuntimeError("depth was insufficient to observe every first escape")

    bounds = linear_bound_candidates(rows, train_max)
    affine = affine_candidates(
        {p: saved[train_max, p] for p in (0, 1)}, train_max,
        depth // 2,
        {p: saved[holdout_max, p] for p in (0, 1)}, holdout_max)
    quadratic_train_support = min(train_max, 8)
    quadratic_holdout_support = min(holdout_max, 12)
    quadratic = quadratic_candidates(
        {p: saved[quadratic_train_support, p] for p in (0, 1)},
        quadratic_train_support, depth // 2,
        {p: saved[quadratic_holdout_support, p] for p in (0, 1)},
        quadratic_holdout_support)
    exact = [[w, max(r["maximum_first_escape_depth"] for r in rows
                     if r["support"] == w)] for w in range(1, holdout_max + 1)]
    return {
        "target": (
            "Prove a uniform finite upper bound on the first forced left one "
            "beyond any finite support radius; this excludes eventual period two."),
        "train_max_support": train_max,
        "holdout_max_support": holdout_max,
        "constructed_depth": depth,
        "rows": rows,
        "combined_escape_depth_by_support": exact,
        "linear_bound_candidates": bounds,
        "affine_spatial_recurrence_candidates": affine,
        "quadratic_spatial_recurrence_search": {
            "train_support": quadratic_train_support,
            "holdout_support": quadratic_holdout_support,
            "maximum_memory": 12,
            "candidates": quadratic,
        },
        "verdict": (
            "Finite exact data only. A linear law surviving holdout is a proof "
            "target, not a theorem. No common affine or quadratic recurrence "
            "means only that these deliberately small induction languages failed."),
        "prize_ceiling": (
            "A proved bound would exclude eventual period two for every finite "
            "Rule 30 configuration. Periods at least three would remain open."),
    }


def verify_artifact(path: Path) -> dict:
    expected = json.loads(path.read_text(encoding="utf-8"))
    if "train_max_support" in expected:
        actual = run(expected["train_max_support"],
                     expected["holdout_max_support"],
                     expected["constructed_depth"])
        ok = actual == expected
        return {"artifact": str(path), "kind": "full-report", "match": ok}
    if "depth" in expected and "rows" in expected:
        mismatches = []
        for saved in expected["rows"]:
            actual, _ = horizon_row(saved["support"], expected["depth"],
                                    saved["phase"])
            if actual != saved:
                mismatches.append({"expected": saved, "actual": actual})
        return {"artifact": str(path), "kind": "support-supplement",
                "match": not mismatches, "mismatches": mismatches}
    raise ValueError("unrecognized invariant artifact schema")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-max", type=int, default=10)
    parser.add_argument("--holdout-max", type=int, default=16)
    parser.add_argument("--depth", type=int, default=64)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        verification = verify_artifact(args.verify)
        print(json.dumps(verification, indent=2))
        return 0 if verification["match"] else 1
    report = run(args.train_max, args.holdout_max, args.depth)
    payload = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8", newline="")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

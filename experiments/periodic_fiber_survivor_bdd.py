#!/usr/bin/env python3
"""Measure exact ROBDD complexity of alternating-fibre survivor sets.

For a right support of width ``w``, let ``S_k`` contain the right words whose
forced left bits at depths ``w+1, ..., w+k`` are all zero.  An alternating
finite fibre exists only if this descending chain never becomes empty.  The
scalar horizon records when it empties; this script keeps the exact Boolean
set and asks whether it has a compact ordered binary decision diagram (ROBDD).

Small ROBDDs are useful proof-search evidence, not a proof: a constant-true
survivor set has the smallest possible diagram and never empties.  Rule 15,
which has a genuine finite alternating fibre, is included as that control.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.periodic_trace_fiber import forced_left_prefix, verify_trace


def robdd_stats(truth: np.ndarray, order: tuple[int, ...]) -> dict:
    """Return exact reduced OBDD sizes for a Boolean truth table.

    Assignment ``q`` is stored at ``truth[q]`` and bit ``v`` of ``q`` is
    variable ``v``.  ``order`` lists variables from root to leaves.  Terminal
    nodes are excluded from ``nonterminal_nodes`` and the per-layer widths.
    """
    values = np.asarray(truth, dtype=np.uint8)
    if values.ndim != 1 or len(values) == 0 or len(values) & (len(values) - 1):
        raise ValueError("truth table length must be a positive power of two")
    variables = len(values).bit_length() - 1
    if tuple(sorted(order)) != tuple(range(variables)):
        raise ValueError("order must be a permutation of all variables")

    # Re-index assignments so that the last variable in the requested order
    # is the least-significant bit.  Adjacent entries can then be reduced from
    # the leaves upward.
    y = np.arange(len(values), dtype=np.uint32)
    q = np.zeros(len(values), dtype=np.uint32)
    for level, variable in enumerate(order):
        q |= ((y >> (variables - 1 - level)) & 1) << variable
    nodes = values[q].astype(np.int64)

    next_id = 2
    widths = [0] * variables
    for level in range(variables - 1, -1, -1):
        pairs = nodes.reshape(-1, 2)
        parents = np.empty(len(pairs), dtype=np.int64)
        reduced = pairs[:, 0] == pairs[:, 1]
        parents[reduced] = pairs[reduced, 0]
        distinct = ~reduced
        if np.any(distinct):
            unique, inverse = np.unique(pairs[distinct], axis=0,
                                        return_inverse=True)
            widths[level] = len(unique)
            parents[distinct] = inverse + next_id
            next_id += len(unique)
        nodes = parents

    return {
        "order": list(order),
        "root_is_terminal": bool(nodes[0] < 2),
        "root_terminal_value": int(nodes[0]) if nodes[0] < 2 else None,
        "nonterminal_nodes": next_id - 2,
        "max_layer_width": max(widths, default=0),
        "layer_widths": widths,
    }


def survivor_profile(support: int, depth: int, phase: int,
                     rule: int = 30) -> dict:
    """Profile the last nonempty exact survivor set for one fibre family."""
    rows, left, target, values = forced_left_prefix(
        depth, support, "01", rotation=phase, rule=rule)
    mismatches = verify_trace(rows, target, rule)
    if mismatches:
        raise RuntimeError(f"constructed fibre failed verification: {mismatches}")

    survivors = np.ones(len(values), dtype=bool)
    last_nonempty = survivors.copy()
    last_k = 0
    counts = []
    escape_depth = None
    for k, column in enumerate(left[:, support:].T, start=1):
        survivors &= column == 0
        count = int(survivors.sum())
        counts.append(count)
        if count:
            last_nonempty = survivors.copy()
            last_k = k
        else:
            escape_depth = support + k
            break

    natural = tuple(range(support))
    reverse = tuple(reversed(natural))
    return {
        "rule": rule,
        "support": support,
        "phase": phase,
        "right_halves": len(values),
        "verification_mismatches": mismatches,
        "escape_depth": escape_depth,
        "right_censored": escape_depth is None,
        "last_nonempty_constraints": last_k,
        "last_nonempty_survivors": int(last_nonempty.sum()),
        "survivor_counts": counts,
        "natural_order_robdd": robdd_stats(last_nonempty, natural),
        "reverse_order_robdd": robdd_stats(last_nonempty, reverse),
    }


def run(max_support: int, depth: int) -> dict:
    if not 1 <= max_support <= 18:
        raise ValueError("max_support must be in 1..18")
    if depth < max_support:
        raise ValueError("depth must be at least max_support")
    profiles = [survivor_profile(w, depth, phase)
                for w in range(1, max_support + 1) for phase in (0, 1)]

    # Rule 15 obeys F(l,c,r)=1 XOR l.  With target 01 and an all-zero left
    # half, the survivor set is constant true at every depth: maximally simple
    # ROBDD structure does not imply eventual emptiness.
    _, left15, target15, values15 = forced_left_prefix(
        depth, 1, "01", rule=15)
    rule15_truth = np.all(left15[:, 1:] == 0, axis=1)
    rule15_control = {
        "trace_mismatches": verify_trace(
            forced_left_prefix(depth, 1, "01", rule=15)[0], target15, 15),
        "survivors_at_depth": int(rule15_truth.sum()),
        "right_halves": len(values15),
        "robdd": robdd_stats(rule15_truth, (0,)),
    }
    return {
        "experiment": "periodic-fiber-survivor-bdd",
        "target": "01",
        "max_support": max_support,
        "depth": depth,
        "profiles": profiles,
        "rule15_finite_alternating_control": rule15_control,
        "claim_limit": (
            "Exact bounded survivor sets and exact ROBDD sizes only. Small or "
            "stable diagrams do not prove a width-independent quotient, and "
            "without a proved ranking or absence of accepting cycles they do "
            "not prove that the survivor chain empties for every support."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-support", type=int, default=16)
    parser.add_argument("--depth", type=int, default=64)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = run(args.max_support, args.depth)
    payload = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8", newline="")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Adversarial complexity audit for Nersissian's Rule 30 support-set method.

This module implements the *explicit* support-set recurrence from the published
binomial-Lucas lifting and instruments every n-dependent construction step.

It deliberately does not claim to implement the paper's masked-dyadic-block
compression. That representation is the object of a later faithful
reconstruction. The purpose here is to establish a transparent cold-start
baseline and make preprocessing impossible to hide behind a warm query time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCE_URL = "https://www.wolframcloud.com/obj/b04b6551-fecf-465d-b02d-63d95abd751c"
DISCUSSION_URL = "https://community.wolfram.com/groups/-/m/t/1802242"


@dataclass
class ConstructionStats:
    target_m: int
    layers_built: int = 0
    or_pairs: int = 0
    or_toggles: int = 0
    linear_toggles: int = 0
    increment_ops: int = 0
    peak_support_size: int = 1
    final_support_size: int = 0
    final_max_index: int = 0


@dataclass
class QueryStats:
    n: int
    support_size: int
    lucas_tests: int
    elapsed_seconds: float
    value: int


def _toggle(target: set[int], value: int) -> None:
    if value in target:
        target.remove(value)
    else:
        target.add(value)


def or_convolution_mod2(
    a: set[int],
    b: set[int],
    stats: ConstructionStats | None = None,
) -> set[int]:
    """OR-convolution over F2: output multiplicities are reduced modulo two."""
    out: set[int] = set()
    for left in a:
        for right in b:
            if stats is not None:
                stats.or_pairs += 1
                stats.or_toggles += 1
            _toggle(out, left | right)
    return out


def next_support(
    prev1: set[int],
    prev2: set[int],
    stats: ConstructionStats | None = None,
) -> set[int]:
    """Published Rule 30 recurrence.

    S_m = Inc((S_{m-1} * S_{m-2}) delta S_{m-1} delta S_{m-2}),
    with S_1={0}, S_2={1}. Here ``delta`` is symmetric difference and ``*``
    is OR-convolution with multiplicities reduced modulo two.
    """
    merged = or_convolution_mod2(prev1, prev2, stats)
    for value in prev1:
        if stats is not None:
            stats.linear_toggles += 1
        _toggle(merged, value)
    for value in prev2:
        if stats is not None:
            stats.linear_toggles += 1
        _toggle(merged, value)
    if stats is not None:
        stats.increment_ops += len(merged)
    return {value + 1 for value in merged}


def build_support(target_m: int) -> tuple[set[int], ConstructionStats]:
    """Construct S_target_m from scratch and return deterministic work counts."""
    if target_m < 1:
        raise ValueError("target_m must be >= 1")
    stats = ConstructionStats(target_m=target_m)
    if target_m == 1:
        support = {0}
        stats.final_support_size = 1
        return support, stats

    prev2 = {0}  # S_1
    prev1 = {1}  # S_2
    if target_m == 2:
        stats.final_support_size = 1
        stats.final_max_index = 1
        return prev1, stats

    for _m in range(3, target_m + 1):
        cur = next_support(prev1, prev2, stats)
        stats.layers_built += 1
        stats.peak_support_size = max(stats.peak_support_size, len(cur))
        prev2, prev1 = prev1, cur

    stats.final_support_size = len(prev1)
    stats.final_max_index = max(prev1, default=0)
    return prev1, stats


def lucas_binomial_parity(n: int, r: int) -> int:
    """Return C(n,r) mod 2 via Lucas' theorem."""
    if n < 0 or r < 0 or r > n:
        return 0
    return int((r & ~n) == 0)


def evaluate_from_support(support: Iterable[int], n: int) -> QueryStats:
    """Evaluate b(m,n) from an already-materialised explicit support set."""
    values = tuple(support)
    start = time.perf_counter()
    parity = 0
    for r in values:
        parity ^= lucas_binomial_parity(n, r)
    elapsed = time.perf_counter() - start
    return QueryStats(
        n=n,
        support_size=len(values),
        lucas_tests=len(values),
        elapsed_seconds=elapsed,
        value=parity,
    )


def center_bit_cold(n: int) -> tuple[int, ConstructionStats, QueryStats]:
    """Compute c_n from n alone using the explicit published recurrence."""
    if n < 0:
        raise ValueError("n must be non-negative")
    support, construction = build_support(n + 1)
    query = evaluate_from_support(support, n)
    return query.value, construction, query


class SequentialSupport:
    """Reusable state exposing the difference between warm and cold queries."""

    def __init__(self) -> None:
        self.supports: dict[int, set[int]] = {1: {0}, 2: {1}}
        self.max_m = 2

    def ensure(self, target_m: int) -> ConstructionStats:
        if target_m < 1:
            raise ValueError("target_m must be >= 1")
        stats = ConstructionStats(target_m=target_m)
        if target_m <= self.max_m:
            support = self.supports[target_m]
            stats.final_support_size = len(support)
            stats.final_max_index = max(support, default=0)
            stats.peak_support_size = len(support)
            return stats

        for m in range(self.max_m + 1, target_m + 1):
            cur = next_support(self.supports[m - 1], self.supports[m - 2], stats)
            self.supports[m] = cur
            stats.layers_built += 1
            stats.peak_support_size = max(stats.peak_support_size, len(cur))
        self.max_m = target_m
        support = self.supports[target_m]
        stats.final_support_size = len(support)
        stats.final_max_index = max(support, default=0)
        return stats

    def center_bit(self, n: int) -> tuple[int, ConstructionStats, QueryStats]:
        if n < 0:
            raise ValueError("n must be non-negative")
        construction = self.ensure(n + 1)
        query = evaluate_from_support(self.supports[n + 1], n)
        return query.value, construction, query


def independent_center_bits(n_steps: int) -> list[int]:
    """Use the repo's independently cross-checked integer Rule 30 engine."""
    from prize_lab import center_bits_int

    return center_bits_int(n_steps, margin=64)


def validate(max_n: int) -> dict:
    """Validate every c_n through max_n against the existing Rule 30 engine."""
    expected = independent_center_bits(max_n)
    mismatches: list[dict] = []
    state = SequentialSupport()
    for n in range(max_n + 1):
        got, _, _ = state.center_bit(n)
        if got != expected[n]:
            mismatches.append({"n": n, "got": got, "expected": expected[n]})
            break
    return {
        "max_n": max_n,
        "checked": max_n + 1 if not mismatches else mismatches[0]["n"] + 1,
        "ok": not mismatches,
        "mismatches": mismatches,
    }


def audit_n(n: int) -> dict:
    cold_start = time.perf_counter()
    value, construction, query = center_bit_cold(n)
    cold_elapsed = time.perf_counter() - cold_start
    return {
        "n": n,
        "value": value,
        "cold_elapsed_seconds": cold_elapsed,
        "construction": asdict(construction),
        "warm_explicit_query": asdict(query),
        "accounting_note": (
            "warm_explicit_query scans explicit S_(n+1); it is not the paper's "
            "masked-dyadic O(log n) evaluator"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-values",
        default="4,8,12,16,20",
        help="comma-separated center indices; explicit supports can grow quickly",
    )
    parser.add_argument("--validate-up-to", type=int, default=24)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    n_values = [int(part.strip()) for part in args.n_values.split(",") if part.strip()]
    if not n_values or any(n < 0 for n in n_values):
        raise SystemExit("--n-values must contain non-negative integers")
    if args.validate_up_to < 0:
        raise SystemExit("--validate-up-to must be non-negative")

    validation = validate(args.validate_up_to)
    if not validation["ok"]:
        raise RuntimeError(f"support recurrence failed validation: {validation}")

    sequential = SequentialSupport()
    sequential_rows = []
    for n in sorted(n_values):
        value, construction, query = sequential.center_bit(n)
        sequential_rows.append(
            {
                "n": n,
                "value": value,
                "new_construction": asdict(construction),
                "query": asdict(query),
            }
        )

    out = {
        "experiment": "nersissian_end_to_end_audit",
        "source": {
            "paper_cloud": SOURCE_URL,
            "contest_discussion": DISCUSSION_URL,
            "scope": (
                "explicit support recurrence plus Lucas evaluation; masked dyadic "
                "compression is not reconstructed here"
            ),
        },
        "validation": validation,
        "cold_isolated": [audit_n(n) for n in n_values],
        "sequential_reuse": sequential_rows,
        "interpretation_guardrail": (
            "These measurements characterize this explicit construction. They do "
            "not prove an asymptotic lower bound for all implementations."
        ),
    }
    payload = json.dumps(out, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8", newline="")
    print(payload, end="")


if __name__ == "__main__":
    main()

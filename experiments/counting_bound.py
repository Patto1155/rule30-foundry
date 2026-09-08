"""Finite-prefix counting bound for shortcut-class search experiments.

A negative result ("nothing in model class M fits the first n bits") is
informative only when log2|M| >= n. Below that threshold every sequence produces
the same negative and the experiment has measured |M|, not the sequence.

See docs/theory/finite-prefix-counting-bound.md.

Usage:
    python experiments/counting_bound.py --pretty
    python experiments/counting_bound.py --base 2 --max-states 24
    python experiments/counting_bound.py --verdict 5:128 --verdict 8:128
"""

from __future__ import annotations

import argparse
import json
import math
import sys


def log2_dfao_upper(states: int, base: int) -> float:
    """log2 UPPER bound on distinct s-state base-b binary DFAO behaviours.

    s^(s*b) transition tables times 2^s output labelings. Every behaviour is
    realised by at least one syntactic automaton, so this is a valid upper bound.
    Use this for vacuity verdicts: overstating |M| makes the vacuity claim
    conservative.
    """
    if states < 1:
        raise ValueError("states must be >= 1")
    return states * base * math.log2(states) + states


def log2_dfao_lower(states: int, base: int) -> float:
    """log2 LOWER bound on distinct s-state base-b binary DFAO behaviours.

    Quotient the syntactic count by the (s-1)! relabelings of the non-initial
    states. By Burnside the number of orbits is at least |X|/|G|, so this is a
    LOWER bound, not an upper one -- automata with nontrivial stabilizers have
    short orbits and only increase the orbit count.

    Use this for fair-threshold questions ("how many states before a random
    sequence could plausibly be fitted"): understating |M| demands more states
    and is therefore conservative in that direction.
    """
    return log2_dfao_upper(states, base) - math.lgamma(states) / math.log(2)


def log2_lfsr_behaviours(order: int, n_bits: int) -> float:
    """log2 expected fits for GF(2) LFSRs of given order on n bits, plus n.

    An order-L LFSR is pinned by its feedback polynomial (2^L choices) once the
    first L bits are taken as state; the remaining n-L bits each match with
    probability 1/2. Expected fits = 2^(2L-n), so the informative threshold is
    L >= n/2.
    """
    return 2.0 * order - n_bits


def dfao_table(base: int, max_states: int) -> list[dict]:
    rows = []
    for s in range(1, max_states + 1):
        up = log2_dfao_upper(s, base)
        lo = log2_dfao_lower(s, base)
        rows.append(
            {
                "states": s,
                "log2_behaviours_upper": round(up, 3),
                "log2_behaviours_lower": round(lo, 3),
                "max_informative_n_bits": int(math.floor(up)),
                "log2_exhaustive_space": round(up, 1),
            }
        )
    return rows


def fair_thresholds(base: int, n_list: list[int], search_cap: int = 4096) -> list[dict]:
    out = []
    for n in n_list:
        s_opt = next((s for s in range(1, search_cap)
                      if log2_dfao_upper(s, base) >= n), None)
        s_cons = next((s for s in range(1, search_cap)
                       if log2_dfao_lower(s, base) >= n), None)
        out.append(
            {
                "n_bits": n,
                "states_required_optimistic": s_opt,
                "states_required_conservative": s_cons,
                "log2_exhaustive_space": (
                    round(log2_dfao_upper(s_opt, base), 1) if s_opt else None
                ),
            }
        )
    return out


def verdict(states: int, n_bits: int, base: int) -> dict:
    log2_m = log2_dfao_upper(states, base)   # upper bound => conservative vacuity
    margin = log2_m - n_bits
    return {
        "states": states,
        "n_bits": n_bits,
        "base": base,
        "log2_behaviours_upper": round(log2_m, 3),
        "log2_expected_fits": round(margin, 3),
        "informative": bool(margin >= 0),
        "p_random_sequence_also_has_no_fit_at_least": (
            1.0 - 2.0**margin if margin < 0 else None
        ),
        "reading": (
            "informative: a negative result here constrains the sequence"
            if margin >= 0
            else f"VACUOUS: any sequence gives this negative (expected fits 2^{margin:.1f})"
        ),
    }


def parse_verdict(spec: str) -> tuple[int, int]:
    try:
        a, b = spec.split(":")
        return int(a), int(b)
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(
            f"--verdict expects STATES:NBITS, got {spec!r}"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=int, default=2, help="digit base b (default 2)")
    ap.add_argument("--max-states", type=int, default=25)
    ap.add_argument(
        "--n-bits",
        default="64,128,256,512,1024",
        help="comma-separated prefix lengths for the fair-threshold table",
    )
    ap.add_argument(
        "--verdict",
        action="append",
        type=parse_verdict,
        default=None,
        metavar="STATES:NBITS",
        help="evaluate a specific claim, e.g. --verdict 5:128",
    )
    ap.add_argument("--pretty", action="store_true", help="human table on stderr")
    args = ap.parse_args(argv)

    n_list = [int(x) for x in args.n_bits.split(",") if x.strip()]
    verdicts = args.verdict or [(4, 128), (5, 128), (8, 128), (8, 256)]

    out = {
        "artifact_type": "rule30.counting_bound",
        "base": args.base,
        "dfao_table": dfao_table(args.base, args.max_states),
        "fair_thresholds": fair_thresholds(args.base, n_list),
        "verdicts": [verdict(s, n, args.base) for s, n in verdicts],
        "reference": "docs/theory/finite-prefix-counting-bound.md",
    }

    if args.pretty:
        w = sys.stderr
        w.write(f"\nDFAO behaviour counts, base b={args.base}\n")
        w.write(f"{'s':>4} | {'log2|M|':>9} | {'max informative n':>18}\n")
        w.write("-" * 38 + "\n")
        for r in out["dfao_table"]:
            w.write(
                f"{r['states']:>4} | {r['log2_behaviours_upper']:>9.1f} | "
                f"{r['max_informative_n_bits']:>18}\n"
            )
        w.write("\nFair thresholds (smallest s making n informative)\n")
        for r in out["fair_thresholds"]:
            w.write(
                f"  n={r['n_bits']:>5} -> s >= {r['states_required_optimistic']}"
                f" (optimistic bound) .. {r['states_required_conservative']}"
                f" (conservative bound)\n"
            )
        w.write("\nVerdicts\n")
        for v in out["verdicts"]:
            w.write(f"  s<={v['states']}, n={v['n_bits']}: {v['reading']}\n")
        w.write("\n")
        w.flush()

    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


def log2_annihilator_space(window: int, degree: int) -> float:
    """log2 of the number of GF(2) polynomials of degree <= d in w variables.

    The monomial basis has D = sum_{k<=d} C(w,k) elements, so the class has
    2^D members. D itself is the number the rank test cares about; it is
    returned separately by `annihilator_dimension`.
    """
    return float(annihilator_dimension(window, degree))


def annihilator_dimension(window: int, degree: int) -> int:
    """D = sum_{k=0..d} C(w,k): the monomial-matrix column count."""
    return sum(math.comb(window, k) for k in range(degree + 1))


def max_zeros_of_degree(window: int, degree: int) -> int:
    """Most zeros a NONZERO degree-<=d polynomial in w variables can have.

    The minimum distance of the Reed-Muller code RM(d, w) is 2^(w-d), so any
    nonzero f of degree <= d takes the value 1 at least 2^(w-d) times and
    therefore vanishes at most 2^w - 2^(w-d) times. This is a theorem, not an
    estimate, and it is what makes the annihilator gate exact.
    """
    if degree >= window:
        return (1 << window) - 1
    return (1 << window) - (1 << (window - degree))


def annihilator_verdict(window: int, degree: int, n_distinct: int,
                        margin_bits: int = 64) -> dict:
    """Is an annihilator search at these parameters informative?

    This model class fails in *both* directions, unlike the DFAO class, so
    there are two independent gates.

    Forced-negative gate. An annihilator must vanish on every observed window,
    so the observed windows all lie in its zero set. By the Reed-Muller bound
    above, a nonzero degree-<=d polynomial has at most 2^w - 2^(w-d) zeros. So
    if we observe MORE than that many distinct windows, no such polynomial can
    exist -- whatever the sequence is. "No annihilator found" is then a
    restatement of "the stream is varied enough", and says nothing about Rule
    30. Measured on the golden 10M stream this rules out w <= 22 at d = 2, and
    w <= 18 is the extreme case where the windows cover GF(2)^w outright.

    Forced-positive gate. The monomial matrix has D columns. Fewer than D
    independent rows leaves a kernel by dimension alone, so an "annihilator"
    would be an artifact of sampling. Requiring n_distinct >= D + margin_bits
    puts the chance that random data clears the bar at about 2^-margin_bits.

    The window between the two gates is where the experiment can say something:
    few enough distinct windows that a relation is not excluded by counting,
    and many more than D so that finding one is not automatic.
    """
    dim = annihilator_dimension(window, degree)
    max_zeros = max_zeros_of_degree(window, degree)
    covers = n_distinct >= (1 << window)
    surplus = n_distinct - (dim + margin_bits)

    if n_distinct > max_zeros:
        informative, reading = False, (
            f"VACUOUS: {n_distinct} distinct windows exceeds the {max_zeros} "
            f"zeros a nonzero degree-{degree} polynomial in {window} variables "
            "can have (Reed-Muller bound), so no annihilator can exist for any "
            "sequence. This negative is forced."
            + (" The windows cover GF(2)^w outright." if covers else "")
        )
    elif surplus < 0:
        informative, reading = False, (
            f"VACUOUS: {n_distinct} distinct windows against D={dim} monomials "
            f"leaves a kernel by dimension alone (need >= {dim + margin_bits}). "
            "Any sequence gives this positive."
        )
    else:
        informative, reading = True, (
            f"informative: D={dim} monomials, {n_distinct} distinct windows -- "
            f"{surplus} beyond the D+{margin_bits} bar, and {max_zeros - n_distinct} "
            "below the Reed-Muller ceiling. Full rank here is a real negative; "
            "a confirmed kernel vector is a real shortcut candidate."
        )

    return {
        "window_bits": window,
        "degree": degree,
        "monomial_dimension": dim,
        "log2_class_size": round(log2_annihilator_space(window, degree), 3),
        "max_zeros_reed_muller": max_zeros,
        "n_distinct_windows": n_distinct,
        "covers_window_space": bool(covers),
        "margin_bits": margin_bits,
        "surplus_rows": surplus,
        "headroom_below_ceiling": max_zeros - n_distinct,
        "informative": informative,
        "reading": reading,
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

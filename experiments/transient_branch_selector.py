"""Test small transient invariants for choosing a zero-word successor.

At a settled zero word the periodic-word map has two solutions.  The exact
diagonal recursion says which one occurs: the last transient 1 in the zero
diagonal is the final reset.  This probe asks whether its phase is a compact
selector on every early zero word reachable by the existing simulation.

The phase is small; obtaining it may still require an ever longer transient.
We therefore report both selector accuracy and the absolute last-reset time.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


PERIODS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def simulate_selected(diagonals, steps):
    """Return exact D_d(t), t=0..steps, for selected diagonal indices."""
    if not diagonals or min(diagonals) < 0 or steps < 0:
        raise ValueError("need nonnegative steps and diagonal indices")
    wanted = sorted(set(diagonals))
    width = 2 * steps + 3
    centre = steps + 1
    mask = (1 << width) - 1
    window_mask = (1 << (max(wanted) + 1)) - 1
    state = 1 << centre
    columns = {d: bytearray(steps + 1) for d in wanted}
    for t in range(steps + 1):
        window = (state >> (centre - t)) & window_mask
        for d in wanted:
            columns[d][t] = (window >> d) & 1
        state = ((state << 1) ^ (state | (state >> 1))) & mask
    return columns


def minimal_tail_period(column, candidates=PERIODS, repeats=4):
    for period in candidates:
        span = repeats * period
        if len(column) >= span and all(
                column[t] == column[t - period]
                for t in range(len(column) - span + period, len(column))):
            return period
    return None


def periodic_word(column, period):
    word = 0
    start = len(column) - period
    for t in range(start, len(column)):
        word |= int(column[t]) << (t % period)
    return word


def branch_words(predecessor, period):
    """The two period-p solutions for input (predecessor, zero)."""
    x = 0
    word0 = 0
    for phase in range(period):
        word0 |= x << phase
        x ^= (predecessor >> phase) & 1
    if x:
        raise ValueError("odd predecessor has no period-p successor")
    return word0, word0 ^ ((1 << period) - 1)


def predict_from_last_reset(predecessor, period, last_one):
    """Try to choose using only the settled predecessor and reset phase."""
    if last_one is None:
        # The zero diagonal was zero from the origin, so D_(z+1)(0)=0 is the
        # boundary condition and no transient reset exists.
        t_phase, x = 0, 0
    else:
        phase = last_one % period
        t_phase = (phase + 1) % period
        x = 1 ^ ((predecessor >> phase) & 1)
    word = 0
    for _ in range(period):
        word |= x << t_phase
        x ^= (predecessor >> t_phase) & 1
        t_phase = (t_phase + 1) % period
    return word


def predict_from_tail_state(predecessor, period, start_time, start_value):
    """Continue from one bit after both inputs have reached their tails."""
    phase = start_time % period
    x = start_value
    word = 0
    for _ in range(period):
        word |= x << phase
        x ^= (predecessor >> phase) & 1
        phase = (phase + 1) % period
    return word


def last_mismatch(column, word, period):
    bad = [t for t, bit in enumerate(column)
           if bit != ((word >> (t % period)) & 1)]
    return bad[-1] if bad else None


def simple_hypotheses(case):
    """Deliberately tiny selectors which do not inspect the transient."""
    z = case["zero_word_at_d"]
    u = case["predecessor_word"]
    p = case["analysis_period"]
    return {
        "constant_0": 0,
        "constant_1": 1,
        "zero_d_parity": z & 1,
        "zero_d_block_parity": (z // p) & 1,
        "predecessor_phase0": u & 1,
        "predecessor_half_popcount_parity":
            (u & ((1 << max(1, p // 2)) - 1)).bit_count() & 1,
    }


def analyze(zero_words, steps):
    if zero_words != sorted(set(zero_words)) or any(z < 2 for z in zero_words):
        raise ValueError("zero words must be unique, sorted, and >=2")
    targets = [d for z in zero_words for d in (z - 1, z, z + 1)]
    started = time.perf_counter()
    columns = simulate_selected(targets, steps)
    cases = []
    for z in zero_words:
        a, b, x = columns[z - 1], columns[z], columns[z + 1]
        p_before = minimal_tail_period(a)
        p_zero = minimal_tail_period(b)
        p_after = minimal_tail_period(x)
        if None in (p_before, p_zero, p_after) or p_zero != 1 or any(b[-4:]):
            raise ValueError(f"event d={z} is not settled in the simulated horizon")
        period = max(p_before, p_after)
        u = periodic_word(a, period)
        actual = periodic_word(x, period)
        candidates = branch_words(u, period)
        if actual not in candidates:
            raise ValueError(f"actual successor at d={z} is not a map candidate")
        ones = [t for t, bit in enumerate(b) if bit]
        last_one = ones[-1] if ones else None
        predecessor_mismatch = last_mismatch(a, u, period)
        cutoff = max(last_one if last_one is not None else -1,
                     predecessor_mismatch if predecessor_mismatch is not None else -1)
        start_time = cutoff + 1
        last_reset_prediction = predict_from_last_reset(u, period, last_one)
        tail_prediction = predict_from_tail_state(
            u, period, start_time, int(x[start_time]))
        case = {
            "zero_word_at_d": z,
            "period_before": p_before,
            "period_after": p_after,
            "analysis_period": period,
            "predecessor_word": u,
            "candidates": list(candidates),
            "actual_successor": actual,
            "actual_candidate_index": candidates.index(actual),
            "last_one_time": last_one,
            "last_one_phase": last_one % period if last_one is not None else None,
            "last_reset_invariant_bits": math.ceil(math.log2(period + 1)),
            "predecessor_last_mismatch": predecessor_mismatch,
            "successor_last_mismatch": last_mismatch(x, actual, period),
            "last_reset_prediction": last_reset_prediction,
            "last_reset_prediction_correct": last_reset_prediction == actual,
            "tail_cutoff_time": cutoff,
            "tail_start_phase": start_time % period,
            "tail_start_value": int(x[start_time]),
            "tail_state_invariant_bits":
                (math.ceil(math.log2(period)) if period > 1 else 0) + 1,
            "tail_state_prediction": tail_prediction,
            "tail_state_prediction_correct": tail_prediction == actual,
            "dependency_time_over_diagonal":
                round(cutoff / z, 6) if cutoff >= 0 else 0.0,
        }
        case["simple_predictions"] = simple_hypotheses(case)
        cases.append(case)

    split = max(1, len(cases) * 2 // 3)
    names = list(cases[0]["simple_predictions"])
    scored = []
    for name in names:
        train_ok = sum(c["simple_predictions"][name] == c["actual_candidate_index"]
                       for c in cases[:split])
        valid_ok = sum(c["simple_predictions"][name] == c["actual_candidate_index"]
                       for c in cases[split:])
        scored.append({"selector": name,
                       "train_correct": train_ok, "train_total": split,
                       "validation_correct": valid_ok,
                       "validation_total": len(cases) - split})
    dependencies = [c["last_one_time"] or 0 for c in cases]
    cutoff_dependencies = [c["tail_cutoff_time"] for c in cases]
    ratios = [c["dependency_time_over_diagonal"] for c in cases]
    return {
        "artifact_type": "rule30.transient_branch_selector",
        "artifact_version": 1,
        "scope": "all zero words in the existing exact d<1M event artifact",
        "params": {"steps": steps, "zero_words": zero_words,
                   "chronological_training_cases": split,
                   "chronological_validation_cases": len(cases) - split},
        "cases": cases,
        "small_settled_only_selectors": scored,
        "last_reset_phase_selector": {
            "correct": sum(c["last_reset_prediction_correct"] for c in cases),
            "total": len(cases),
            "maximum_stored_bits":
                max(c["last_reset_invariant_bits"] for c in cases),
            "verdict": "insufficient when the predecessor is still transient",
        },
        "tail_phase_plus_bit_selector": {
            "training_correct":
                sum(c["tail_state_prediction_correct"] for c in cases[:split]),
            "training_total": split,
            "validation_correct":
                sum(c["tail_state_prediction_correct"] for c in cases[split:]),
            "validation_total": len(cases) - split,
            "maximum_stored_bits":
                max(c["tail_state_invariant_bits"] for c in cases),
        },
        "dependency_growth": {
            "last_one_times": dependencies,
            "tail_cutoff_times": cutoff_dependencies,
            "maximum_tail_cutoff": max(cutoff_dependencies),
            "last_two_cutoff_increase":
                cutoff_dependencies[-1] - cutoff_dependencies[-2],
            "ratio_range": [min(ratios), max(ratios)],
            "compression_verdict":
                "phase plus one tail-boundary bit is sufficient, but no bounded way to compute that bit from settled words was found; exact dependency time grows with d",
        },
        "stop_reason":
            "no direct simulation beyond the computationally reachable early events",
        "elapsed_s": round(time.perf_counter() - started, 3),
    }


def load_zero_words(path):
    artifact = json.loads(path.read_text(encoding="utf-8"))
    return artifact["zero_words"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path,
                        default=Path("data/wedge/period16_walk_1e6.json"))
    parser.add_argument("--steps", type=int, default=130_000)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    result = analyze(load_zero_words(args.events), args.steps)
    if args.verify:
        expected = json.loads(args.verify.read_text(encoding="utf-8"))
        # Wall time is deliberately not part of the scientific payload.
        result["elapsed_s"] = expected.get("elapsed_s")
        if result != expected:
            raise ValueError("artifact differs from complete recomputation")
        print(json.dumps({"verified": True, "cases": len(result["cases"])}, indent=2))
        return
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n",
                            encoding="utf-8", newline="")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

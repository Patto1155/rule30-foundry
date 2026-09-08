"""Finite-horizon driven half-line probe; this is NOT the single-seed orbit.

Prescribing the center replaces its actual dynamics. Finite positive support
is retained, but compatibility with the left half-line is not imposed.
Suffix periods are observations on a finite window, never asymptotic claims.
"""
import argparse
from collections import Counter
import json
from pathlib import Path


def rows(initial, boundary, steps):
    """Positive-half rows at times 0..steps; low bit is cell 1."""
    if (type(initial) is not int or type(steps) is not int or initial < 0
            or steps < 0 or not boundary or set(boundary) - set("01")):
        raise ValueError("need nonnegative initial/steps and nonempty binary boundary")
    width = initial.bit_length() + steps + 1
    mask = (1 << width) - 1
    row = initial
    yield row
    for t in range(steps):
        row = (((row << 1) | int(boundary[t % len(boundary)]))
               ^ (row | (row >> 1))) & mask
        yield row


def trace(initial, boundary, steps):
    """Return positive cell 1 at times 0..steps."""
    return [row & 1 for row in rows(initial, boundary, steps)]


def loop_completion_report(certificate):
    """Test a specific finite right completion of the two loop witnesses.

    Keep their common initial strip's positive cells, set all cells farther
    right to zero, and impose their common center word. Only a necessary
    right-half/center consistency condition is checked, not the left dynamics.
    """
    if __package__:
        from .periodic_strip_graph import verify_output_loops
    else:
        from periodic_strip_graph import verify_output_loops
    verify_output_loops(certificate)
    width, word = certificate["width"], certificate["center_word"]
    radius = width // 2
    phase, root = certificate["closed_walks"][0][0]
    boundary = word[phase:] + word[:phase]
    initial = root >> (radius + 1)
    length = certificate["block_length"]
    evolution = list(rows(initial, boundary, length))
    required_left = [int(boundary[(t+1) % len(word)])
                     ^ (int(boundary[t % len(word)]) | (evolution[t] & 1))
                     for t in range(length)]
    outcomes = []
    for choice, path in enumerate(certificate["closed_walks"]):
        edge = next((t for t, (_, row) in enumerate(path)
                     if ((row >> (width-1)) & 1)
                     != ((evolution[t] >> (radius-1)) & 1)), None)
        half = next((t for t, (_, row) in enumerate(path)
                     if (row >> (radius+1))
                     != (evolution[t] & ((1 << radius)-1))), None)
        left = next((t for t in range(length)
                     if required_left[t] != certificate["output_blocks"][choice][t]), None)
        outcomes.append({"choice": choice, "first_right_edge_mismatch": edge,
                         "first_positive_strip_mismatch": half,
                         "first_required_left_mismatch": left,
                         "right_completion_compatible": half is None and left is None})
    # Different left blocks cannot both result from one deterministic right
    # continuation and the same center trace.
    assert sum(o["right_completion_compatible"] for o in outcomes) <= 1
    return {"width": width, "center_word": word, "block_length": length,
            "finite_right_initial": initial, "choices": outcomes,
            "scope": "specific zero-exterior completion; neither full seed nor all completions"}


def naive_trace(initial, boundary, steps):
    """Independent truth-table reference with an ample zero right halo."""
    width = initial.bit_length() + steps + 1
    row = [(initial >> i) & 1 for i in range(width)]
    out = [row[0]]
    for t in range(steps):
        padded = [int(boundary[t % len(boundary)])] + row + [0]
        row = [(30 >> (4*padded[i] + 2*padded[i+1] + padded[i+2])) & 1
               for i in range(width)]
        out.append(row[0])
    return out


def seed_snapshots(last_onset=64):
    """Independent full-line single-seed rows before applying the forcing."""
    row = {0: 1}
    for t in range(last_onset + 1):
        yield t, sum(row.get(i, 0) << (i-1) for i in range(1, t+1)), row.get(0, 0)
        row = {i: (30 >> (4*row.get(i-1, 0) + 2*row.get(i, 0)
                         + row.get(i+1, 0))) & 1 for i in range(-t-1, t+2)}


def suffix_report(bits, max_period):
    """Exact periods within [floor(horizon/2), horizon], inclusive."""
    start = (len(bits)-1)//2
    if max_period < 1 or 2*max_period > len(bits)-start:
        raise ValueError("suffix must contain at least two copies of largest period")
    periods = [p for p in range(1, max_period+1)
               if all(bits[t] == bits[t-p] for t in range(start+p, len(bits)))]
    p = periods[0] if periods else None
    defects = [] if p is None else [t for t in range(p, len(bits)) if bits[t] != bits[t-p]]
    return {"smallest_suffix_period": p, "suffix_start": start,
            "last_defect_time": max(defects, default=None)}


def scan(steps=2048, max_period=64):
    families = {f"all_8bit_boundary_{b}": [(i, i, b) for i in range(256)]
                for b in ("0", "1", "01", "10")}
    families["seed_snapshot_alternating"] = [(t, row, "01" if c == 0 else "10")
                                            for t, row, c in seed_snapshots()]
    result = {"scope": "driven right half-line, not full seed orbit; finite suffix only",
              "steps": steps, "max_period": max_period, "families": {}}
    for name, cases in families.items():
        histogram = Counter()
        examples = []
        latest = None
        for label, initial, boundary in cases:
            bits = trace(initial, boundary, steps)
            report = suffix_report(bits, max_period)
            period = report["smallest_suffix_period"]
            histogram[str(period)] += 1
            last = report["last_defect_time"]
            if last is not None:
                latest = last if latest is None else max(latest, last)
            if period is None and len(examples) < 3:
                examples.append({"case": label, "initial": initial, "boundary": boundary,
                                 "prefix64": "".join(map(str, bits[:64])),
                                 "suffix64": "".join(map(str, bits[-64:]))})
        result["families"][name] = {"cases": len(cases), "period_histogram": dict(histogram),
                                    "latest_defect_among_resolved": latest,
                                    "unresolved_examples": examples}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scan", action="store_true", help="run the bounded family scan")
    mode.add_argument("--check-loops", type=Path)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--max-period", type=int, default=64)
    args = parser.parse_args()
    result = ([loop_completion_report(c) for c in json.loads(args.check_loops.read_text())]
              if args.check_loops else scan(args.steps, args.max_period))
    print(json.dumps(result, indent=2))

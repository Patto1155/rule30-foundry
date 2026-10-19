#!/usr/bin/env python
"""Constraints forced by an alternating centre trace, and one exact falsifier.

Rule 30: a(t+1,i) = a(t,i-1) XOR (a(t,i) OR a(t,i+1)). Write l = a(.,-1),
c = a(.,0), r = a(.,1), r2 = a(.,2). Suppose the centre trace alternates,
c_{t+1} = 1 XOR c_t. Then, directly from the recurrence:

  D1  c_t = 1  =>  l_t = 1                      the OR latches; right half irrelevant
  D2  c_t = 0  =>  l_t = 1 XOR r_t
  D3  a(t,-2) = r_{t+1} if c_t = 1, else r_t    column -2 mirrors column +1
  D4' c_t = 1  =>  l_{t+1} = r_t OR r2_t
  D5  c_t = 1  =>  r_{t+1} = 1 XOR (r_t OR r2_t)

D1 and D4' together say: **column -1 is identically 1 across an alternating
stretch exactly when (r_t, r2_t) is never (0,0) at a 1-phase inside it.** Half
of "column -1 is eventually constant" is therefore free, and the other half is
one condition.

Why that matters: Condrey (arXiv:2609.09431v1, preprint) Cor. 5 proves no
column of a nonzero finite Rule 30 orbit is eventually constant. So the
candidate lemma

    a nonzero finitely supported row with an eventually alternating centre
    trace has column -1 eventually identically 1

would contradict Cor. 5 and exclude eventual period two, nonconstant, for
every nonzero finite configuration. That is a PARTIAL result: every eventual
period of at least 3 would remain open, and no uniform argument over the
period is known. It is not Prize 1.

**The falsifier this script runs.** Exhaustively over every nonzero row of
support radius w, find the longest alternating centre prefix, then ask whether
the double-zero event (r_t, r2_t) = (0,0) still occurs at a 1-phase inside it.

Read the answer carefully in both directions, because neither is a proof:

  - the event PERSISTS among rows with maximal alternating prefixes: the
    sub-lemma has no local mechanism and cannot be derived from a bounded
    stretch of alternation. Any proof must use infinite alternation together
    with finite support.
  - the event VANISHES as the prefix lengthens: the constraint tightens with
    alternation length, which is the design signal worth a symbolic attempt.

A finite horizon is evidence for designing a lemma, not proof of one, and a
finite-prefix violation is NOT a refutation of an asymptotic lemma: no finite
row alternates forever, so every row here violates the hypothesis eventually.

    python experiments/alternating_trace_constraints.py --pretty
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

from experiments.rule30_open_utils import step_naive_open  # noqa: E402


def step(rows: np.ndarray) -> np.ndarray:
    left = np.zeros_like(rows)
    left[..., 1:] = rows[..., :-1]
    right = np.zeros_like(rows)
    right[..., :-1] = rows[..., 1:]
    return left ^ (rows | right)


def self_test(width: int = 81, steps: int = 40, seed: int = 30) -> dict:
    rng = np.random.default_rng(seed)
    bad = 0
    for _ in range(6):
        a = rng.integers(0, 2, size=width, dtype=np.uint8)
        b = a.copy()
        for _ in range(steps):
            a = step_naive_open(a)
            b = step(b)
            bad += int(np.count_nonzero(a != b))
    return {"rows": 6, "steps": steps, "mismatches": bad}


def scan(w: int, steps: int) -> dict:
    """Every nonzero row of support radius w; columns -1,0,1,2 over time."""
    span = 2 * w + 1
    width = 2 * (w + steps) + 5
    centre = width // 2
    idx = np.arange(1 << span, dtype=np.uint32)
    bits = ((idx[:, None] >> np.arange(span, dtype=np.uint32)[None, :]) & 1).astype(np.uint8)
    bits = bits[bits.any(axis=1)]
    rows = np.zeros((bits.shape[0], width), dtype=np.uint8)
    rows[:, centre - w: centre + w + 1] = bits

    cols = np.empty((4, bits.shape[0], steps + 1), dtype=np.uint8)
    for t in range(steps + 1):
        for j, d in enumerate((-1, 0, 1, 2)):
            cols[j, :, t] = rows[:, centre + d]
        if t < steps:
            rows = step(rows)
    l, c, r, r2 = cols

    # Longest alternating prefix: h = first t with c_t == c_{t+1}, else steps.
    same = c[:, :-1] == c[:, 1:]
    h = np.where(same.any(axis=1), same.argmax(axis=1), steps)

    # Verify the derived identities inside each row's own alternating window,
    # and look for the double-zero event at a 1-phase there.
    # h is the first t with c_t == c_{t+1}, so the trace alternates AT t exactly
    # for t < h. D1 and D2 each read c_{t+1}, so they are claims about t < h and
    # nowhere else -- asserting them at t = h is asserting them outside their
    # own hypothesis, which is not a violation of anything.
    t_idx = np.arange(steps + 1)[None, :]
    inside = t_idx < h[:, None]
    one_phase = inside & (c == 1)
    zero_phase = inside & (c == 0)

    d1_bad = int(np.count_nonzero(one_phase & (l != 1)))
    d2_bad = int(np.count_nonzero(zero_phase & (l != (1 ^ r))))
    # D5 needs no window: it is the recurrence at i = 1 with c_t = 1 substituted,
    # so it holds at every 1-phase whether or not the trace alternates there.
    shifted = np.zeros_like(r)
    shifted[:, :-1] = r[:, 1:]
    d5_bad = int(np.count_nonzero((c[:, :-1] == 1)
                                  & (shifted[:, :-1] != (1 ^ (r | r2))[:, :-1])))

    dz = one_phase & (r == 0) & (r2 == 0)
    has_dz = dz.any(axis=1)

    out = {"radius": w, "configurations": int(bits.shape[0]),
           "identity_violations": {"D1": d1_bad, "D2": d2_bad, "D5": d5_bad},
           "max_alternating_prefix": int(h.max() + 1)}
    by_h = []
    for hv in sorted(set(h.tolist())):
        sel = h == hv
        by_h.append({"prefix_length": int(hv + 1), "rows": int(sel.sum()),
                     "rows_with_double_zero_at_a_1_phase": int(has_dz[sel].sum())})
    out["by_alternating_prefix_length"] = by_h
    top = h == h.max()
    out["at_maximal_prefix"] = {
        "rows": int(top.sum()),
        "with_double_zero_at_a_1_phase": int(has_dz[top].sum())}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-radius", type=int, default=9)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--out", type=Path)
    p.add_argument("--pretty", action="store_true")
    args = p.parse_args()

    check = self_test()
    if check["mismatches"]:
        print("alternating_trace_constraints: stepper disagrees with the naive one; "
              "refusing to report", file=sys.stderr)
        return 2
    if args.self_test:
        print(json.dumps(check, indent=2))
        return 0

    rows = [scan(w, args.steps) for w in range(1, args.max_radius + 1)]
    bad = sum(sum(x["identity_violations"].values()) for x in rows)
    persists = all(x["at_maximal_prefix"]["with_double_zero_at_a_1_phase"] > 0
                   for x in rows)
    report = {
        "steps": args.steps, "self_test": check, "rows": rows,
        "derived_identity_violations_total": bad,
        "double_zero_persists_at_maximal_prefix": bool(persists),
        "verdict": (
            "THE SUB-LEMMA HAS NO LOCAL MECHANISM. The double-zero event "
            "(a(t,1), a(t,2)) = (0,0) at a 1-phase still occurs among the rows "
            "achieving the LONGEST alternating prefix at every radius measured. "
            "So 'column -1 is identically 1' is not forced by any bounded "
            "stretch of alternation, and a proof must use infinite alternation "
            "together with finite support -- not a local argument."
            if persists else
            "The double-zero event disappears at maximal alternating prefix for "
            "at least one radius: the constraint tightens with alternation "
            "length, which is the design signal for a symbolic attempt."),
        "claim_limit": (
            "No period-two exclusion is claimed or proved here. This is a "
            "bounded finding about small support radius, used to design a "
            "lemma. No finite row alternates forever, so every row here "
            "violates the lemma's hypothesis eventually; a finite-prefix "
            "violation is not a refutation. Even if proved, the lemma would "
            "exclude only eventual period two and leave every period at least "
            "three open."),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2) if args.pretty else json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

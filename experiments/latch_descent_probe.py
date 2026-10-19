#!/usr/bin/env python
"""Exact falsifier for the latch-descent obligation (Prize 1).

The obligation
--------------
If the centre column `c` of the single-seed orbit is eventually periodic, is
column -1 eventually periodic too? If so, two *adjacent* columns are eventually
periodic, which the width-2 theorem forbids (Kopra, TCS 946 (2023) 113668,
Thm 3.5, attributed there to Jen (1990) Prop. 3), and Prize 1 follows for every
period at once. The bridge closes with no residue; the lemma is what is open.

Sideways determinism at the origin gives, exactly,

    c_{t+1} = l_t XOR (c_t OR r_t)   so   l_t = c_{t+1} XOR (c_t OR r_t)

with `l_t = a(t,-1)`, `c_t = a(t,0)`, `r_t = a(t,1)`. That splits by phase:

    c_t = 1  ->  l_t = 1 XOR c_{t+1}          pinned by c alone (the OR latches)
    c_t = 0  ->  l_t = c_{t+1} XOR r_t        depends on column +1

So *half* the lemma is free. The whole open content is the 0-phase residue:
if `r` restricted to the 0-phases of `c` is eventually periodic, so is column -1.

What this probe decides
-----------------------
1. **Instrument check.** The identity holds at every t on the real orbit.
   Any violation means the simulator or the indexing is wrong, not the theory.

2. **Pinned fraction.** How much of column -1 the latch gives for free.

3. **Monotone-latch lemma (theorem-shaped).** At i = 1 the rule reads
   `r_{t+1} = c_t XOR (r_t OR a(t,2))`, so whenever `c_t = 0`,
   `r_{t+1} = r_t OR a(t,2) >= r_t`. Column +1 is non-decreasing across every
   step where the centre is 0, and can only fall at a 1-phase of `c`. Checked
   on the seed orbit and exhaustively over every nonzero finite configuration
   of small support radius, where it must never fail.

4. **The decisive one: is the residue a function of a bounded centre window?**
   The cheapest possible proof of the lemma would show `r_t` on 0-phases is
   determined by a window of `c` around t. A single *collision* -- two 0-phase
   times whose centre windows agree while `r` differs -- refutes that for that
   width outright. This is a constructive exclusion, not a search that came
   back empty, so the counting bound does not blunt it.

   It does bound where the probe can look. A collision needs a repeated
   window, so a prefix of N bits can exhibit one only up to its longest
   repeated factor, about 2*log2(N) for a sequence with this one's window
   statistics. Past that every observed window is distinct, any assignment of
   outputs to distinct inputs is realised by some function, and the test has no
   power. That is a property of THIS prefix, not a fact about every sequence --
   a constant sequence repeats its windows at every width. The probe marks
   those rows uninformative rather than clean.

**What this does NOT show.** The exclusion is of one specific predictor shape:
an exact, time-independent function of a symmetric centre window, carrying no
auxiliary state, measured on the ACTUAL orbit. A proof of the lemma gets to
assume the centre column is eventually periodic -- a hypothetical object whose
windows repeat constantly, where this measurement says nothing -- and may use
period-dependent windows, auxiliary state, cells outside the window, or
establish periodicity of column -1 without predicting r at all. So this rules
out an unconditional strengthening that someone might reach for first. It does
not rule out local methods in general, and it must not be reported as doing so.

Controls, because an instrument that cannot detect structure excludes nothing:
a positive control where the residue IS a width-1 function (it must show zero
collisions at W=1), and a random null.

    python experiments/latch_descent_probe.py --pretty
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.rule30_open_utils import step_naive_open  # noqa: E402


def step_fast(row: np.ndarray) -> np.ndarray:
    """Vectorised open-boundary Rule 30 step, checked against the naive one.

    The naive stepper is a Python loop over the whole row, so the cost of a
    T-step run is quadratic *in Python*, which caps T at a few thousand -- and
    the window range this probe can speak about grows like 2*log2(T), so that
    cap is exactly what decides whether the experiment says anything. This is
    the same arithmetic on numpy arrays, and `--self-test` requires the two to
    agree cell for cell before any result is produced.
    """
    left = np.empty_like(row)
    left[0] = 0
    left[1:] = row[:-1]
    right = np.empty_like(row)
    right[-1] = 0
    right[:-1] = row[1:]
    return left ^ (row | right)


def self_test(width: int = 97, steps: int = 60, seed: int = 30) -> dict:
    """The fast stepper against the repo's naive one, on random rows."""
    rng = np.random.default_rng(seed)
    mismatches = 0
    for _ in range(8):
        row = rng.integers(0, 2, size=width, dtype=np.uint8)
        a = row.copy()
        b = row.copy()
        for _ in range(steps):
            a = step_naive_open(a)
            b = step_fast(b)
            mismatches += int(np.count_nonzero(a != b))
    return {"random_rows": 8, "steps": steps, "mismatches": mismatches}


def three_columns(steps: int, margin: int = 4):
    """Columns -1, 0, +1 and +2 of the single-seed orbit, times 0..steps.

    Width is set so the open boundary never reaches the sampled columns inside
    the horizon: the light cone spreads one cell per step in each direction.
    """
    width = 2 * (steps + margin) + 1
    centre = width // 2
    row = np.zeros(width, dtype=np.uint8)
    row[centre] = 1
    out = np.empty((4, steps + 1), dtype=np.uint8)
    for t in range(steps + 1):
        out[0, t] = row[centre - 1]
        out[1, t] = row[centre]
        out[2, t] = row[centre + 1]
        out[3, t] = row[centre + 2]
        if t < steps:
            row = step_fast(row)
    return out


def identity_check(l, c, r):
    """l_t == c_{t+1} XOR (c_t OR r_t) at every t. Exact, no tolerance."""
    t = np.arange(len(c) - 1)
    predicted = c[t + 1] ^ (c[t] | r[t])
    bad = int(np.count_nonzero(predicted != l[t]))
    return {"positions": int(len(t)), "violations": bad,
            "pinned_positions": int(np.count_nonzero(c[t] == 1)),
            "pinned_fraction": round(float(np.mean(c[t] == 1)), 6)}


def monotone_latch_on_orbit(c, r, r2):
    """Whenever c_t = 0, r_{t+1} >= r_t. A one-line consequence of the rule."""
    t = np.arange(len(c) - 1)
    zero = t[c[t] == 0]
    drops = int(np.count_nonzero(r[zero + 1] < r[zero]))
    predicted = r[zero] | r2[zero]
    return {"zero_phase_steps": int(len(zero)), "drops": drops,
            "closed_form_violations": int(np.count_nonzero(predicted != r[zero + 1]))}


def monotone_latch_exhaustive(max_radius: int, steps: int):
    """The same lemma over every nonzero finite configuration of small radius.

    The lemma is seed-blind on purpose: it is a fact about the rule, so it is
    the one part of this card that a seed-blind argument is allowed to carry.
    """
    results = []
    for w in range(1, max_radius + 1):
        width = 2 * (w + steps) + 3
        centre = width // 2
        drops = 0
        configs = 0
        for bits in product((0, 1), repeat=2 * w + 1):
            if not any(bits):
                continue
            configs += 1
            row = np.zeros(width, dtype=np.uint8)
            row[centre - w: centre + w + 1] = bits
            prev_c = prev_r = None
            for _ in range(steps):
                cc, rr = int(row[centre]), int(row[centre + 1])
                if prev_c == 0 and rr < prev_r:
                    drops += 1
                prev_c, prev_r = cc, rr
                row = step_naive_open(row)
        results.append({"radius": w, "configurations": configs, "drops": drops})
    return results


def window_collisions(c, r, widths, max_positions=None):
    """Is r on the 0-phases of c a function of a centre window of width W?

    Window at t is c[t-W+1 .. t+W-1] -- symmetric and causal-agnostic, because
    the lemma is about eventual periodicity, not about computing anything.
    A collision is two 0-phase times with equal windows and unequal r.
    """
    n = len(c)
    out = []
    for w in widths:
        seen = {}
        collisions = 0
        distinct = 0
        checked = 0
        witnesses = []
        lo, hi = w - 1, n - w
        for t in range(lo, hi):
            if c[t]:
                continue
            if max_positions and checked >= max_positions:
                break
            checked += 1
            key = c[t - w + 1: t + w].tobytes()
            bit = int(r[t])
            if key not in seen:
                seen[key] = (bit, t)
                distinct += 1
            elif seen[key][0] != bit:
                collisions += 1
                if len(witnesses) < 3:
                    first_bit, first_t = seen[key]
                    # A witness is the whole certificate: two times, the shared
                    # window written out, and the two differing outputs. Counts
                    # alone are a report; this is checkable by hand.
                    witnesses.append({
                        "t1": first_t, "t2": t,
                        "window": "".join(str(int(b)) for b in c[t - w + 1: t + w]),
                        "r_at_t1": first_bit, "r_at_t2": bit})
        # A collision needs a repeated window. Once every observed window is
        # distinct the test is powerless -- any assignment of outputs to
        # distinct inputs is realised by SOME function -- so a clean answer
        # there is a property of this prefix's window statistics, not evidence
        # about the sequence. It is NOT "forced for every sequence": a constant
        # sequence repeats its windows at every width.
        repeats = checked - distinct
        out.append({"window_bits": 2 * w - 1, "half_width": w,
                    "zero_phase_positions": checked,
                    "distinct_windows": distinct, "repeated_windows": repeats,
                    "collisions": collisions, "witnesses": witnesses,
                    "informative": bool(repeats > 0),
                    "verdict": ("REFUTES any function of the %d-bit window"
                                % (2 * w - 1))
                               if collisions else
                               ("no collision, but UNINFORMATIVE: no window "
                                "repeats in this prefix, so the test has no "
                                "power here"
                                if repeats == 0 else
                                "no collision at this width (informative)")})
    return out


def controls(n, widths, seed):
    """Positive control and null, so a clean answer can be told from a blind one."""
    rng = np.random.default_rng(seed)
    c = rng.integers(0, 2, size=n + 1, dtype=np.uint8)
    # Two positive controls, and the second one is the sharper calibration.
    #
    #   r_t = c_t      is a function of the width-1 window, so EVERY width must
    #                  come back with zero collisions.
    #   r_t = c_{t+1}  is not: the symmetric window c[t-W+1 .. t+W-1] first
    #                  contains c_{t+1} at W=2 (3 bits). So this control must
    #                  collide at 1 bit and then be clean from 3 bits on.
    #
    # The first shows the probe can see a function that is there; the second
    # shows it locates the width correctly instead of merely saying "clean".
    r_null = rng.integers(0, 2, size=n + 1, dtype=np.uint8)
    return {"positive_control_width1": window_collisions(c, c.copy(), widths),
            "positive_control_width3": window_collisions(c, np.roll(c, -1), widths),
            "random_null": window_collisions(c, r_null, widths)}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--widths", type=str, default="1,2,3,4,5,6,7,8,9,10,12")
    p.add_argument("--exhaustive-radius", type=int, default=5)
    p.add_argument("--exhaustive-steps", type=int, default=12)
    p.add_argument("--seed", type=int, default=30)
    p.add_argument("--out", type=Path)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--pretty", action="store_true")
    args = p.parse_args()

    check = self_test()
    if check["mismatches"]:
        print(f"latch_descent_probe: fast stepper disagrees with the naive one "
              f"({check['mismatches']} cells); refusing to report", file=sys.stderr)
        return 2
    if args.self_test:
        print(json.dumps(check, indent=2))
        return 0

    widths = [int(x) for x in args.widths.split(",") if x.strip()]
    l, c, r, r2 = three_columns(args.steps)
    report = {
        "steps": args.steps,
        "self_test": check,
        "identity": identity_check(l, c, r),
        "monotone_latch_orbit": monotone_latch_on_orbit(c, r, r2),
        "monotone_latch_exhaustive": monotone_latch_exhaustive(
            args.exhaustive_radius, args.exhaustive_steps),
        "residue_window_collisions": window_collisions(c, r, widths),
        "controls": controls(args.steps, widths, args.seed),
        "repeated_factor_ceiling_bits": round(2 * math.log2(args.steps), 1),
        "claim_limit": (
            "A collision PROVES, constructively, that no function of that window "
            "exists -- on this prefix, for the single seed, for that predictor "
            "shape. The absence of a collision proves nothing once the windows "
            "stop repeating, and the probe marks those rows uninformative. "
            "Neither outcome is an asymptotic statement, and neither excludes "
            "local proof strategies in general: the lemma's hypothesis is a "
            "periodic centre column, which is not the object measured here."),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2) if args.pretty else json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

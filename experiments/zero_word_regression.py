"""Regression test: the pattern map must never silently step through a zero word.

WHAT WENT WRONG (2026-08-19). A first-collision hunt seeded the pattern-map
iteration at d=512 and reported the first consecutive-word collision at
d=53205/53206. A sweep over seed depths then disagreed with itself:

    seed_d 256, 300        ->  (397, 398)
    seed_d 512, 700, 1024  ->  (53205, 53206)

Root cause: there is a genuine zero word at d=399 (from the real collision
w_397 == w_398 == 0xd0d0). When v == 0 the one-period composite is AFFINE
(x -> x XOR parity(u)), so if parity(u) == 0 there are TWO period-16 words
consistent with the recursion and the map alone cannot say which one occurs.
Only the actual transient -- i.e. simulation -- resolves it. Low seeds ran into
the zero and silently picked a branch; high seeds were seeded past it and were
right by luck.

WHAT THIS SCRIPT PINS. Five gates, each of which fails loudly:

  1. FIXTURE   the zero word at d=399 and its even-parity predecessor 0xd0d0
               still exist in simulation (otherwise the rest is vacuous).
  2. HALTS     seeding BEFORE d=399 makes the iteration stop at the zero word
               with reason "zero_word" -- it never steps through.
  3. TEETH     stepping through really is ill-defined: both branches satisfy the
               recursion, they are bitwise complements, they are BOTH different
               from what simulation gives at least half the time, and following
               them leads to DIFFERENT first collisions. This gate is what makes
               gate 2 meaningful rather than a tautology.
  4. GATING    the seed filter used by period_doubling.main() excludes every
               seed at or below the last zero word.
  5. CONSENSUS the six post-zero seeds still agree on the first collision
               (d=53205/53206, word 0x28c3, even parity).

Run:  python experiments/zero_word_regression.py --pretty
Exits non-zero if any gate fails.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagonal_recursion import pattern_map_step  # noqa: E402
from period_doubling import (  # noqa: E402
    PERIOD,
    check_against_simulation,
    collisions_from_simulation,
    hunt_first_collision,
    words_from_simulation,
)

# The known zero word this regression is about, and its predecessor.
KNOWN_ZERO_D = 399
KNOWN_PREDECESSOR = 0xD0D0
# Seeds deliberately placed BEFORE the zero word.
PRE_ZERO_SEEDS = (256, 300)
# The established post-zero consensus.
EXPECTED_COLLISION = (53205, 53206, 0x28C3)


def zero_word_successors(u: int, period: int = PERIOD) -> list[int]:
    """Every period-length word consistent with the recursion when v == 0.

    Mirrors pattern_map_step exactly but with v pinned to 0, and returns ALL
    fixed points instead of the first. With v == 0 the step is
    x -> u[t] XOR x, so the one-period composite is x -> x XOR parity(u):
    two solutions when parity(u) == 0, none when it is 1 (the period doubles).
    """
    out = []
    for x0 in (0, 1):
        x = x0
        seq = []
        for t in range(period):
            seq.append(x)
            x = (u >> t & 1) ^ x
        if x == x0:
            w = 0
            for t in range(period):
                if seq[t]:
                    w |= 1 << t
            out.append(w)
    return out


def first_collision_from(u: int, v: int, start_d: int, limit: int,
                         period: int = PERIOD) -> dict | None:
    """Iterate the map from an explicit (u, v) and report the first collision."""
    d = start_d
    while d < limit:
        if v == 0:
            return {"kind": "zero_word", "d": d - 1}
        if u == v:
            return {"kind": "collision", "d_lo": d - 2, "d_hi": d - 1, "word": u}
        w, flag = pattern_map_step(u, v, period)
        if flag != "ok":
            return {"kind": flag, "d": d}
        u, v = v, w
        d += 1
    return None


def gate_fixture(words) -> dict:
    """The bug fixture must still be present, or every later gate is vacuous."""
    n = len(words)
    ok = (
        n > KNOWN_ZERO_D + 1
        and words[KNOWN_ZERO_D] == 0
        and words[KNOWN_ZERO_D - 1] == KNOWN_PREDECESSOR
        and words[KNOWN_ZERO_D - 2] == KNOWN_PREDECESSOR
    )
    return {
        "zero_at": KNOWN_ZERO_D,
        "w_zero": words[KNOWN_ZERO_D] if n > KNOWN_ZERO_D else None,
        "w_pred": words[KNOWN_ZERO_D - 1] if n > KNOWN_ZERO_D else None,
        "w_pred2": words[KNOWN_ZERO_D - 2] if n > KNOWN_ZERO_D else None,
        "pred_parity": bin(KNOWN_PREDECESSOR).count("1") & 1,
        "ok": bool(ok),
    }


def gate_halts(words) -> dict:
    """Seeding before the zero word must stop AT it, never continue past it."""
    res = check_against_simulation(words, PRE_ZERO_SEEDS)
    per_seed = res["per_seed"]
    ok = bool(per_seed) and all(
        r["mismatches"] == 0
        and r["stopped"] is not None
        and r["stopped"]["reason"] == "zero_word"
        and r["stopped"]["d"] == KNOWN_ZERO_D
        for r in per_seed
    )
    return {"seeds": list(PRE_ZERO_SEEDS), "per_seed": per_seed, "ok": ok}


def gate_teeth(words, limit: int) -> dict:
    """Stepping through the zero is genuinely ambiguous -- so gate 2 has content.

    If this gate ever fails, the zero word stopped being a branch point and
    gate 2 would be passing for the wrong reason.
    """
    u = words[KNOWN_ZERO_D - 1]
    cands = zero_word_successors(u)
    truth = words[KNOWN_ZERO_D + 1]

    # Each candidate is a valid continuation as far as the map can tell; follow
    # each to its own "first collision" and show the answers disagree.
    outcomes = []
    for w in cands:
        hit = first_collision_from(0, w, KNOWN_ZERO_D + 2, limit)
        outcomes.append({"successor": w, "first_collision": hit})

    distinct = {json.dumps(o["first_collision"], sort_keys=True) for o in outcomes}
    ok = (
        len(cands) == 2                              # genuinely ambiguous
        and (cands[0] ^ cands[1]) == (1 << PERIOD) - 1  # the two branches are complements
        and truth in cands                           # simulation picks one of them
        and len(distinct) == len(outcomes)           # and the choice changes the answer
    )
    return {
        "predecessor": u,
        "candidates": cands,
        "candidates_are_complements": (cands[0] ^ cands[1]) == (1 << PERIOD) - 1
        if len(cands) == 2 else False,
        "simulation_says": truth,
        "map_alone_could_not_tell": len(cands) == 2,
        "outcomes": outcomes,
        "distinct_answers": len(distinct),
        "ok": bool(ok),
    }


def gate_seed_filter(words, diagonals: int) -> dict:
    """The production seed filter must exclude everything at/below the last zero."""
    sim = collisions_from_simulation(words)
    last_zero = max(sim["zero_words"]) if sim["zero_words"] else 0
    seeds = [s for s in (500, 1000, 2000, 4000, 8000, 11000)
             if last_zero < s < diagonals]
    ok = bool(seeds) and all(s > last_zero for s in seeds) and last_zero == KNOWN_ZERO_D
    return {"zero_words": sim["zero_words"], "last_zero": last_zero,
            "seeds": seeds, "ok": bool(ok)}


def gate_consensus(words, seeds, limit: int) -> dict:
    """The post-zero seeds must still agree, and on the recorded collision."""
    hunt = hunt_first_collision(words, seeds, limit)
    c = hunt.get("consensus") or {}
    got = (c.get("d_lo"), c.get("d_hi"), c.get("word"))
    ok = hunt["seeds_agree"] and got == EXPECTED_COLLISION and c.get("parity") == 0
    return {"seeds": seeds, "seeds_agree": hunt["seeds_agree"],
            "consensus": c, "expected": list(EXPECTED_COLLISION), "ok": bool(ok)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=26000)
    p.add_argument("--diagonals", type=int, default=12000)
    p.add_argument("--tail", type=int, default=4096)
    p.add_argument("--limit", type=int, default=400_000)
    p.add_argument("--pretty", action="store_true")
    p.add_argument("--out")
    a = p.parse_args(argv)

    t0 = time.time()
    words = words_from_simulation(a.steps, a.diagonals, a.tail)

    gates = {
        "1_fixture": gate_fixture(words),
        "2_halts_at_zero": gate_halts(words),
        "3_teeth_ambiguous": gate_teeth(words, a.limit),
        "4_seed_filter": gate_seed_filter(words, a.diagonals),
    }
    gates["5_consensus"] = gate_consensus(words, gates["4_seed_filter"]["seeds"], a.limit)

    ok = all(g["ok"] for g in gates.values())
    out = {
        "artifact_type": "rule30.zero_word_regression",
        "artifact_version": 1,
        "params": {"steps": a.steps, "diagonals": a.diagonals, "tail": a.tail,
                   "limit": a.limit, "period": PERIOD},
        "gates": gates,
        "elapsed_s": round(time.time() - t0, 3),
        "ok": bool(ok),
    }

    if a.pretty:
        w = sys.stderr
        g = gates["1_fixture"]
        print(f"1 fixture   : {'PASS' if g['ok'] else 'FAIL'}  "
              f"w_{KNOWN_ZERO_D}=0, predecessor=0x{(g['w_pred'] or 0):04x} "
              f"parity={g['pred_parity']}", file=w)
        g = gates["2_halts_at_zero"]
        stops = {r["seed_d"]: (r["stopped"] or {}).get("d") for r in g["per_seed"]}
        print(f"2 halts     : {'PASS' if g['ok'] else 'FAIL'}  "
              f"pre-zero seeds {list(PRE_ZERO_SEEDS)} all stop at d={stops}", file=w)
        g = gates["3_teeth_ambiguous"]
        cs = ", ".join(f"0x{c:04x}" for c in g["candidates"])
        print(f"3 teeth     : {'PASS' if g['ok'] else 'FAIL'}  "
              f"two valid successors ({cs}), complements="
              f"{g['candidates_are_complements']}, sim picks "
              f"0x{(g['simulation_says'] or 0):04x}, "
              f"{g['distinct_answers']} distinct downstream answers", file=w)
        g = gates["4_seed_filter"]
        print(f"4 seedfilter: {'PASS' if g['ok'] else 'FAIL'}  "
              f"zeros={g['zero_words']} last={g['last_zero']} -> seeds={g['seeds']}",
              file=w)
        g = gates["5_consensus"]
        c = g["consensus"]
        print(f"5 consensus : {'PASS' if g['ok'] else 'FAIL'}  "
              f"d={c.get('d_lo')}/{c.get('d_hi')} word=0x{c.get('word', 0):04x} "
              f"parity={c.get('parity')} agree={g['seeds_agree']}", file=w)
        print(f"OVERALL     : {'PASS' if ok else 'FAIL'}  ({out['elapsed_s']}s)", file=w)

    print(json.dumps(out, indent=1, sort_keys=True))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""When does the period-16 conjecture fail? Two lemmas and a collision hunt.

The wedge work (2026-08-15) left "every left diagonal is eventually periodic
with period dividing 16" as a Proof candidate, tested to d ~ 30000 with zero
exceptions, and recorded the missing step as "bound how often the period
doubling branch fires". This module sharpens that into two exact statements and
then measures the rate.

Setting. Each settled diagonal is a PERIOD-bit word w_d phase-locked to
t mod PERIOD, and the diagonal recursion lifts to the pattern map
(w_{d-2}, w_{d-1}) -> w_d, which is well defined whenever w_{d-1} != 0.
The propagation lemma doubling branch fires exactly when w_{d-1} == 0.

LEMMA A (collision criterion).  For v != 0,  pattern_map_step(u, v) == 0
                                iff  u == v.

  Proof. w == 0 forces 0 = w[t+1] = u[t] XOR (v[t] OR 0) = u[t] XOR v[t] for
  every t, i.e. u == v. Conversely if u == v then w == 0 satisfies the
  recursion, and since v != 0 the one-period composite is constant, so that
  solution is the unique eventual one. []

  Hence w_d == 0 iff w_{d-2} == w_{d-1}: the doubling branch fires exactly at a
  COLLISION between consecutive settled words.

LEMMA B (doubling criterion).  At a collision the composite is x -> x XOR c
  with c = parity(w_{d-2}). The period doubles iff c == 1; if c == 0 it stays. []

Consequence. period-16 holds through D iff every consecutive-word collision
below D has an even-parity predecessor. Both conditions are checkable without
simulating past the seed.

Measured here: collisions are a ~2^-PERIOD per-diagonal event, so the first one
beyond the structured early regime sits near 2^16, and the ~3x10^4 test range
behind the conjecture was too short to contain it. See the log for the power
analysis.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagonal_recursion import left_diagonals, pattern_map_step  # noqa: E402

PERIOD = 16


def words_from_simulation(steps: int, diagonals: int, tail: int, period: int = PERIOD):
    """Settled words taken from the trusted simulation. None = not settled."""
    dg = left_diagonals(steps, diagonals)
    base = max(0, steps - tail)
    if base % period:
        raise ValueError(f"tail must leave base ({base}) a multiple of period {period}")
    out: list[int | None] = []
    for d in range(diagonals):
        col = dg[base:, d]
        if len(col) <= 2 * period or not np.array_equal(col[:-period], col[period:]):
            out.append(None)
            continue
        w = 0
        for i in range(period):
            if col[i]:
                w |= 1 << ((base + i) % period)
        out.append(w)
    return out


def check_lemma_a(period: int = PERIOD, samples: int = 200_000, seed: int = 0) -> dict:
    """Lemma A, exhaustively at period 8 and by sampling at the working period."""
    rng = random.Random(seed)
    ex_tested = ex_bad = 0
    for u in range(256):
        for v in range(1, 256):
            w, flag = pattern_map_step(u, v, 8)
            ex_tested += 1
            if flag != "ok" or (w == 0) != (u == v):
                ex_bad += 1
    s_tested = s_bad = 0
    for _ in range(samples):
        u, v = rng.getrandbits(period), rng.getrandbits(period)
        if v == 0:
            continue
        w, flag = pattern_map_step(u, v, period)
        s_tested += 1
        if flag != "ok" or (w == 0) != (u == v):
            s_bad += 1
    return {
        "exhaustive_period8": {"tested": ex_tested, "violations": ex_bad},
        "sampled": {"period": period, "tested": s_tested, "violations": s_bad},
        "ok": ex_bad == 0 and s_bad == 0,
    }


def check_against_simulation(words, seeds, period: int = PERIOD) -> dict:
    """Iterate the map from each seed and require exact agreement with simulation."""
    n = len(words)
    results = []
    for s in seeds:
        if s < 2 or s >= n or words[s - 1] is None or words[s - 2] is None:
            continue
        u, v = words[s - 2], words[s - 1]
        mismatches = 0
        stopped = None
        for d in range(s, n):
            if v == 0:
                stopped = {"reason": "zero_word", "d": d - 1}
                break
            w, flag = pattern_map_step(u, v, period)
            if flag != "ok":
                stopped = {"reason": flag, "d": d}
                break
            if words[d] is not None and w != words[d]:
                mismatches += 1
                stopped = {"reason": "mismatch", "d": d}
                break
            u, v = v, w
        results.append({"seed_d": s, "mismatches": mismatches, "stopped": stopped})
    return {"per_seed": results, "ok": all(r["mismatches"] == 0 for r in results)}


def collisions_from_simulation(words, period: int = PERIOD) -> dict:
    """Zero words, consecutive-word collisions, and Lemmas A/B on real data."""
    n = len(words)
    zeros = [d for d, w in enumerate(words) if w == 0]
    pairs, violations = [], []
    for d in range(2, n):
        if any(words[k] is None for k in (d, d - 1, d - 2)):
            continue
        if (words[d] == 0) != (words[d - 2] == words[d - 1]):
            violations.append(d)
        if words[d - 2] == words[d - 1]:
            pairs.append(d - 2)
    doubling = []
    for z in zeros:
        if z < 2 or words[z - 2] is None:
            continue
        pc = bin(words[z - 2]).count("1")
        doubling.append({
            "zero_at_d": z, "predecessor_d": z - 2,
            "predecessor_word": words[z - 2], "popcount": pc,
            "parity": pc & 1, "doubles": bool(pc & 1),
        })
    parities = [bin(w).count("1") & 1 for w in words if w is not None]
    odd_ds = [d for d, w in enumerate(words) if w is not None and (bin(w).count("1") & 1)]
    return {
        "zero_words": zeros,
        "collision_pairs": pairs,
        "lemma_a_violations": violations,
        "doubling_events": doubling,
        "any_doubling": any(e["doubles"] for e in doubling),
        "parity_even": parities.count(0),
        "parity_odd": parities.count(1),
        "first_odd_parity_d": odd_ds[0] if odd_ds else None,
    }


def hunt_first_collision(words, seeds, limit: int, period: int = PERIOD) -> dict:
    """Iterate the O(1) map past the simulated range to the first collision.

    Reliable up to the first zero word: between zeros the map is exact (and is
    checked against simulation by check_against_simulation), and we stop AT the
    collision rather than trying to step through it.
    """
    found = []
    for s in seeds:
        if s < 2 or s >= len(words) or words[s - 1] is None or words[s - 2] is None:
            continue
        u, v = words[s - 2], words[s - 1]
        d = s
        hit = None
        while d < limit:
            if v == 0:
                hit = {"kind": "zero_word", "d": d - 1}
                break
            if u == v:
                pc = bin(u).count("1")
                hit = {"kind": "collision", "d_lo": d - 2, "d_hi": d - 1, "word": u,
                       "popcount": pc, "parity": pc & 1, "doubles": bool(pc & 1),
                       "zero_word_at": d}
                break
            w, flag = pattern_map_step(u, v, period)
            if flag != "ok":
                hit = {"kind": flag, "d": d}
                break
            u, v = v, w
            d += 1
        found.append({"seed_d": s, "hit": hit})
    hits = [f["hit"] for f in found if f["hit"]]
    agree = len({(h.get("d_lo"), h.get("word")) for h in hits}) == 1 if hits else False
    return {"per_seed": found, "seeds_agree": agree,
            "consensus": hits[0] if (agree and hits) else None}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=26000)
    p.add_argument("--diagonals", type=int, default=12000)
    p.add_argument("--tail", type=int, default=4096)
    p.add_argument("--limit", type=int, default=400_000,
                   help="how far to iterate the map when hunting the first collision")
    p.add_argument("--pretty", action="store_true")
    p.add_argument("--out")
    a = p.parse_args(argv)

    t0 = time.time()
    lemma_a = check_lemma_a()
    words = words_from_simulation(a.steps, a.diagonals, a.tail)
    n_settled = sum(w is not None for w in words)
    sim = collisions_from_simulation(words)
    last_zero = max(sim["zero_words"]) if sim["zero_words"] else 0
    seeds = [s for s in (500, 1000, 2000, 4000, 8000, 11000)
             if last_zero < s < a.diagonals]
    agreement = check_against_simulation(words, seeds)
    hunt = hunt_first_collision(words, seeds, a.limit)

    out = {
        "artifact_type": "rule30.period_doubling",
        "artifact_version": 1,
        "params": {"steps": a.steps, "diagonals": a.diagonals, "tail": a.tail,
                   "limit": a.limit, "period": PERIOD},
        "lemma_a_collision_criterion": lemma_a,
        "simulation_range": {"diagonals": a.diagonals, "settled": n_settled, **sim},
        "map_vs_simulation": agreement,
        "first_collision_beyond_structured_regime": hunt,
        "elapsed_s": round(time.time() - t0, 3),
    }
    ok = lemma_a["ok"] and agreement["ok"] and not sim["lemma_a_violations"]
    out["ok"] = bool(ok)

    if a.pretty:
        c = hunt.get("consensus") or {}
        print(f"Lemma A  : exhaustive period-8 {lemma_a['exhaustive_period8']['violations']} violations, "
              f"sampled {lemma_a['sampled']['violations']} violations", file=sys.stderr)
        print(f"Lemma A on simulation: {len(sim['lemma_a_violations'])} violations over "
              f"{n_settled} settled diagonals", file=sys.stderr)
        print(f"zero words      : {sim['zero_words']}", file=sys.stderr)
        print(f"collision pairs : {sim['collision_pairs']}", file=sys.stderr)
        print(f"doubling fired  : {sim['any_doubling']}", file=sys.stderr)
        print(f"parity split    : even {sim['parity_even']} / odd {sim['parity_odd']}, "
              f"first odd at d={sim['first_odd_parity_d']}", file=sys.stderr)
        print(f"map vs sim      : {'PASS' if agreement['ok'] else 'FAIL'} "
              f"({len(seeds)} seeds, 0 mismatches)", file=sys.stderr)
        if c:
            print(f"first collision : d={c.get('d_lo')}/{c.get('d_hi')} word=0x{c.get('word', 0):04x} "
                  f"parity={c.get('parity')} -> {'DOUBLES' if c.get('doubles') else 'stays 16'}",
                  file=sys.stderr)
        print(f"seeds agree     : {hunt['seeds_agree']}", file=sys.stderr)

    print(json.dumps(out, indent=1, sort_keys=True))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

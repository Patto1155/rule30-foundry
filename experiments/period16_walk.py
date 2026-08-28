"""Decide period-16 by direct simulation: every settled word, every collision.

WHY NOT THE PATTERN-MAP WALK. The plan had been to iterate the 16-bit pattern
map to d ~ 10^6 and resolve each zero word by local simulation. But the branch
points are an artifact of the map's compression, not of the problem: the map
throws away the diagonal transient, and at a zero word the transient is exactly
what disambiguates. Simulating the cone directly has NO branch points -- every
settled word is ground truth.

WHY NOT left_diagonals. It is fast (big-int bitwise ops advance ALL diagonals at
once) but materializes a (T+1, D) array: 1.34 TB at the scale needed here. Its
simulation is only O(T) memory though; the array is the problem.

WHAT THIS DOES. Runs the same big-int CA, but records only a PACKED periodic
tail (default 512 rows = 32 periods, D/8 bytes each). Then:

  settled(d)  <-  rows[i] == rows[i+16] for every recorded i   (bitwise, packed)
  w_d         <-  the 16 phase-locked bits, base chosen = 0 mod 16
  zero words  <-  settled d with w_d == 0
  collisions  <-  settled consecutive d with w_{d-2} == w_{d-1}   (Lemma A)
  VERDICT     <-  parity of each collision predecessor            (Lemma B)

An odd-parity collision refutes period-16 outright. It also directly checks
16-periodicity of every settled diagonal, which is a stronger statement than the
map iteration could make.

Since settle(d) ~ 1.3389*d - 62, T must exceed ~1.34*D for the deepest diagonals
to have settled. Diagonals that have NOT settled by T are reported and excluded
from the verdict rather than being silently counted as clean -- "not settled
within T" is right-censored (AGENTS.md).

Run:  python experiments/period16_walk.py --diagonals 200000 --pretty
Exits non-zero if the verdict is a refutation or if gates fail.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

PERIOD = 16


def walk(diagonals: int, steps: int | None = None, keep: int = 512,
         settle_slope: float = 1.3389, margin: float = 1.10) -> dict:
    """Simulate to T, record a packed periodic tail, extract every settled word."""
    if steps is None:
        steps = int(diagonals * settle_slope * margin) + 4096
    # base (first recorded row) must be a multiple of PERIOD
    base = steps - keep + 1
    base -= base % PERIOD
    keep = steps - base + 1

    width = 2 * steps + 3
    centre = steps + 1
    mask = (1 << width) - 1
    wmask = (1 << diagonals) - 1
    nbytes = (diagonals + 7) // 8

    t0 = time.perf_counter()
    state = 1 << centre
    rows: list[np.ndarray] = []
    for t in range(steps + 1):
        if t >= base:
            win = (state >> (centre - t)) & wmask
            rows.append(np.frombuffer(win.to_bytes(nbytes, "little"), dtype=np.uint8))
        state = ((state << 1) ^ (state | (state >> 1))) & mask
    sim_s = time.perf_counter() - t0

    # minimal period of every diagonal, tested on packed bits.
    # NOTE: this does NOT assume 16. Assuming it is what hid the refutation --
    # a period-32 diagonal simply fails the 16 test and looks "unsettled".
    t0 = time.perf_counter()
    cands = [c for c in (1, 2, 4, 8, 16, 32, 64, 128, 256) if len(rows) > 2 * c]
    period_of = np.zeros(diagonals, dtype=np.int32)      # 0 = not settled at any candidate
    for c in reversed(cands):                            # large -> small, so small wins
        diff = np.zeros(nbytes, dtype=np.uint8)
        for i in range(len(rows) - c):
            diff |= rows[i] ^ rows[i + c]
        ok_c = np.unpackbits(~diff, bitorder="little", count=diagonals).astype(bool)
        period_of[ok_c] = c

    settled = period_of > 0
    settled16 = period_of == PERIOD

    # w_d over the first PERIOD recorded rows (base % PERIOD == 0 -> bit i is phase i)
    words = np.zeros(diagonals, dtype=np.int64)
    for i in range(PERIOD):
        bits = np.unpackbits(rows[i], bitorder="little", count=diagonals)
        words |= bits.astype(np.int64) << i
    extract_s = time.perf_counter() - t0

    n_settled = int(settled.sum())
    unsettled = np.flatnonzero(~settled)
    first_unsettled = int(unsettled[0]) if unsettled.size else diagonals

    hist = {int(c): int((period_of == c).sum()) for c in cands}
    hist[0] = int((period_of == 0).sum())

    # where the period first exceeds 16, and every subsequent increase
    over16 = np.flatnonzero(settled & (period_of > PERIOD))
    first_over16 = int(over16[0]) if over16.size else None

    # Doubling events, tested against Lemma B at the ACTUAL minimal period.
    # Padding a period-p word (p<16) to 16 bits doubles its popcount, so its
    # 16-bit parity is always even -- using that would wrongly predict "never
    # doubles" for every p<16 diagonal. Restrict to p bits (p divides 16).
    zero_ds = np.flatnonzero((period_of == 1) & (words == 0))
    events = []
    for z in zero_ds.tolist():
        if z < 2 or z + 1 >= diagonals:
            continue
        p_prev = int(period_of[z - 1])
        p_next = int(period_of[z + 1])
        if p_prev == 0 or p_next == 0:
            continue
        wp = int(words[z - 2]) & ((1 << p_prev) - 1)
        pc = bin(wp).count("1")
        predicted = 2 * p_prev if (pc & 1) else p_prev
        events.append({
            "zero_word_at_d": z,
            "collision_pair": [z - 2, z - 1],
            "period_before": p_prev,
            "period_after": p_next,
            "predecessor_word_p_bits": wp,
            "predecessor_hex": f"0x{wp:0{max(1, p_prev // 4)}x}",
            "popcount_at_period": pc,
            "parity": pc & 1,
            "lemma_b_predicted_period": predicted,
            "lemma_b_confirmed": p_next == predicted,
        })
    lemma_b_failures = [e for e in events if not e["lemma_b_confirmed"]]

    # Lemma A and collisions, restricted to the region where period is still 16
    ok3 = settled16[2:] & settled16[1:-1] & settled16[:-2]
    zero_d = np.flatnonzero((period_of == 1) & (words == 0))
    coll_ok = settled16[:-2] & settled16[1:-1]
    rhs = words[:-2] == words[1:-1]
    coll_idx = np.flatnonzero(coll_ok & rhs)
    collisions = []
    for c in coll_idx.tolist():
        pred = int(words[c]); pc = int(bin(pred).count("1"))
        collisions.append({"d_lo": c, "d_hi": c + 1, "zero_at": c + 2,
                           "word": pred, "hex": f"0x{pred:04x}",
                           "popcount": pc, "parity": pc & 1,
                           "doubles": bool(pc & 1)})
    bad = [c for c in collisions if c["doubles"]]
    lemma_a_viol = np.flatnonzero(ok3 & ((words[2:] == 0) != rhs)) + 2
    parities = np.array([bin(int(x)).count("1") & 1 for x in words[settled16]],
                        dtype=np.int8)

    return {
        "params": {"diagonals": diagonals, "steps": steps, "keep_rows": keep,
                   "base_row": base, "period": PERIOD},
        "timing": {"simulate_s": round(sim_s, 2), "extract_s": round(extract_s, 2)},
        "settled": {
            "count": n_settled,
            "fraction": round(n_settled / diagonals, 6),
            "first_unsettled_d": first_unsettled,
            "period_histogram": hist,
            "first_d_with_period_over_16": first_over16,
            "censoring_note": "period 0 = not periodic at any tested candidate within "
                              "the recorded tail; right-censored, excluded from verdicts",
        },
        "doubling_events": events,
        "n_doubling_events": len(events),
        "lemma_b_failures": lemma_b_failures,
        "lemma_a_violations": lemma_a_viol.tolist()[:16],
        "zero_words": zero_d.tolist(),
        "n_zero_words": int(zero_d.size),
        "collisions": collisions,
        "n_collisions": len(collisions),
        "odd_parity_collisions": bad,
        "parity_even": int((parities == 0).sum()),
        "parity_odd": int((parities == 1).sum()),
        "verdict": ("REFUTED: period exceeds 16 at d=%d" % first_over16)
                   if first_over16 is not None else
                   "period-16 holds for every settled diagonal tested",
        "ok": (first_over16 is None and lemma_a_viol.size == 0
               and not lemma_b_failures),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--diagonals", type=int, default=200_000)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--keep", type=int, default=512)
    p.add_argument("--pretty", action="store_true")
    p.add_argument("--out")
    a = p.parse_args(argv)

    t0 = time.time()
    res = walk(a.diagonals, a.steps, a.keep)
    res["artifact_type"] = "rule30.period16_walk"
    res["artifact_version"] = 1
    res["elapsed_s"] = round(time.time() - t0, 2)

    if a.pretty:
        w = sys.stderr
        pr, st = res["params"], res["settled"]
        print(f"walk        : d < {pr['diagonals']}, T = {pr['steps']}, "
              f"{pr['keep_rows']} recorded rows", file=w)
        print(f"timing      : simulate {res['timing']['simulate_s']}s, "
              f"extract {res['timing']['extract_s']}s", file=w)
        print(f"settled     : {st['count']} / {pr['diagonals']} "
              f"({st['fraction']:.4f})", file=w)
        print(f"periods     : {st['period_histogram']}", file=w)
        print(f"first d with period > 16 : {st['first_d_with_period_over_16']}", file=w)
        ev = res["doubling_events"]
        print(f"Lemma B     : {len(ev)} events, "
              f"{len(res['lemma_b_failures'])} failures", file=w)
        for e in ev[-6:]:
            print(f"   zero d={e['zero_word_at_d']}: period {e['period_before']} -> "
                  f"{e['period_after']}  w={e['predecessor_hex']} "
                  f"popcount={e['popcount_at_period']} parity={e['parity']} "
                  f"-> predicted {e['lemma_b_predicted_period']} "
                  f"({'OK' if e['lemma_b_confirmed'] else 'FAIL'})", file=w)
        print(f"Lemma A     : {len(res['lemma_a_violations'])} violations", file=w)
        print(f"zero words  : {res['n_zero_words']} -> "
              f"{res['zero_words'][:24]}{' ...' if res['n_zero_words'] > 24 else ''}",
              file=w)
        print(f"collisions  : {res['n_collisions']}", file=w)
        for c in res["collisions"]:
            flag = "ODD -> DOUBLES" if c["doubles"] else "even -> stays 16"
            print(f"   d={c['d_lo']}/{c['d_hi']}  w={c['hex']}  "
                  f"popcount={c['popcount']}  {flag}", file=w)
        print(f"parity      : even {res['parity_even']} / odd {res['parity_odd']}",
              file=w)
        print(f"VERDICT     : {res['verdict']}", file=w)

    print(json.dumps(res, indent=1, sort_keys=True, default=str))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True, default=str),
                               encoding="utf-8")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Walk the pattern map at period 32 to find the predicted 32 -> 64 doubling.

THE OPEN PREDICTION. `docs/CLAIM_LEDGER.md` records `period(d) ~ 2*log2(d)`,
rounded down to a power of two, as a Robust observation, with one stated way to
promote it: "Confirm the 32->64 event predicted near d ~ 8.6e9. Needs the
generalized pattern map, not more direct simulation (T ~ 1.2e10)." Direct
simulation is out: the tail-recording walk in `period16_walk.py` needs
T ~ 1.34*D, so D ~ 8.6e9 means T ~ 1.2e10 rows of a 2.3e10-bit integer.

WHY THE MAP WALK IS AFFORDABLE HERE, HAVING BEEN REJECTED BEFORE.
`period16_walk.py` rejected the map walk because the map is *partial*: at a
zero word two words satisfy the recursion and only the diagonal transient picks
one. That objection is decisive when branch points are common. At period 32
they are not. Working out the algebra of

    x[t+1] = u[t] XOR (v[t] OR x[t])

over one period:

  * where v[t] = 1, x[t+1] = NOT u[t], which is constant in x[t]. So if v != 0
    the one-period composite is a constant map and w_d is UNIQUE -- no branch,
    no dependence on the transient.
  * where v == 0 identically, x[p] = x[0] XOR parity_p(u). Odd parity over the
    full p bits gives no period-p solution at all, so the period exceeds p.
    Even parity gives two solutions, and only then does the walk branch.

That is Lemma B, re-derived. The consequence is the point: **branch points and
doubling events are the same collisions, separated only by parity**, and a
collision is a ~2^-32 per-diagonal event. So the walk to d ~ 10^10 expects only
a couple of collisions total, and roughly half of those are the event we are
hunting. We therefore do not need to resolve branches at all -- we FOLLOW BOTH,
and report whether the conclusion is branch-independent. If every surviving
branch doubles inside the same window, the transient we cannot compute did not
matter.

TWO PARITIES, TWO QUESTIONS -- DO NOT CONFLATE THEM. A period-q word (q < 32)
is 32/q copies of a q-bit block, so its 32-bit popcount is 32/q times the true
one. Lemma B's "does the diagonal's period double?" must therefore be asked at
the MINIMAL period; asking it over 32 bits reports "never doubles" for every
q < 32 diagonal, which is the defect behind the false "doubling never fired"
claim (`docs/experiment-logs/2026-08-19-period16-refuted.md`).

But this walk asks a different question -- "is there still a period-32
solution?" -- and that one is decided by parity over the FULL 32 bits. Ground
truth at the known 16->32 event pins the distinction: at d=87866 the
predecessor is u = 0x28a828a8, minimal period 16, parity(0x28a8) = 5 ODD, so
Lemma B correctly predicts doubling, and indeed w_87867 = 0xcf3030cf has
minimal period 32. Yet parity over all 32 bits is 10, EVEN -- a period-32
solution exists, and the walk must pass through the event rather than halt on
it. Full-32 parity is odd only when the minimal period is already 32, which is
exactly the 32->64 event this hunts. Using minimal-period parity as the halt
condition would stop the walk at the 16->32 doubling and report it as terminal.

GATES (all must pass or the run exits non-zero):
  1. Lemma A on the seeded regime: w_d == 0 iff w_{d-2} == w_{d-1}.
  2. The map, iterated from the seed, reproduces the simulated words exactly
     over the whole overlap -- 0 mismatches, not "mostly agrees".
  3. The seed pair is in a stretch whose measured minimal period really is 32.

Run:  python experiments/pattern_map_walk.py --pretty
      python experiments/pattern_map_walk.py --max-d 12000000000 --pretty
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    from numba import njit
    NUMBA = True
except ImportError:  # pragma: no cover - environment guard
    NUMBA = False

    def njit(*a, **k):  # type: ignore[misc]
        def deco(fn):
            return fn
        return deco if not a or not callable(a[0]) else a[0]

REPO = Path(__file__).resolve().parent.parent
PERIOD = 32
FLAG_OK, FLAG_BRANCH, FLAG_DOUBLE = 0, 1, 2


# --------------------------------------------------------------------------
# the map
# --------------------------------------------------------------------------

@njit(cache=True)
def _minimal_period(w, p):
    """Smallest divisor q of p (a power of two) with w q-periodic."""
    q = 1
    while q < p:
        blk = w & ((np.uint64(1) << np.uint64(q)) - np.uint64(1))
        ok = True
        i = 1
        while i < p // q:
            if ((w >> np.uint64(i * q)) & ((np.uint64(1) << np.uint64(q))
                                           - np.uint64(1))) != blk:
                ok = False
                break
            i += 1
        if ok:
            return q
        q *= 2
    return p


@njit(cache=True)
def _step(u, v, p):
    """(w_{d-2}, w_{d-1}) -> (w_d, flag)."""
    if v == np.uint64(0):
        # Solvability is decided by parity over the FULL p bits, not by
        # Lemma B's minimal-period parity. The two answer different questions:
        #   minimal-period parity -> "does the diagonal's period double?"
        #   full-p parity         -> "is there still a period-p solution?"
        # Checked against ground truth at the known 16->32 event (d=87866):
        # u = 0x28a828a8 has minimal period 16 with parity(0x28a8) = 5, ODD, so
        # Lemma B correctly predicts doubling -- and w_87867 = 0xcf3030cf does
        # have minimal period 32. But parity over all 32 bits is 10, EVEN, so a
        # period-32 solution still exists and the 32-bit walk must continue
        # through the event, not halt on it. Using minimal-period parity here
        # would have stopped the walk at the 16->32 doubling and reported it as
        # terminal. Full-p parity is odd only when the minimal period is
        # already p, which is exactly the 32->64 event this hunts.
        par = 0
        for t in range(p):
            par ^= int((u >> np.uint64(t)) & np.uint64(1))
        if par:
            return np.uint64(0), FLAG_DOUBLE
        return np.uint64(0), FLAG_BRANCH
    # v != 0 -> the one-period composite is constant; one pass finds it
    x = 0
    for t in range(p):
        x = int((u >> np.uint64(t)) & np.uint64(1)) ^ (
            int((v >> np.uint64(t)) & np.uint64(1)) | x)
    w = np.uint64(0)
    for t in range(p):
        if x:
            w |= np.uint64(1) << np.uint64(t)
        x = int((u >> np.uint64(t)) & np.uint64(1)) ^ (
            int((v >> np.uint64(t)) & np.uint64(1)) | x)
    return w, FLAG_OK


@njit(cache=True)
def _walk(u0, v0, p, max_steps):
    """Advance until a branch/doubling or max_steps. Returns (i, flag, u, v)."""
    u = u0
    v = v0
    for i in range(max_steps):
        w, flag = _step(u, v, p)
        if flag != FLAG_OK:
            return i, flag, u, v
        u = v
        v = w
    return max_steps, -1, u, v


@njit(cache=True)
def _replay(u0, v0, p, out):
    """Fill out[i] with w_{d0+i}. Returns steps completed before any flag."""
    u = u0
    v = v0
    for i in range(out.shape[0]):
        w, flag = _step(u, v, p)
        if flag != FLAG_OK:
            return i
        out[i] = w
        u = v
        v = w
    return out.shape[0]


def branch_words(u: int, p: int) -> tuple[int, int]:
    """The two period-p solutions when v == 0 and parity(u) is even.

    x[t+1] = u[t] XOR x[t], so w[t] = x0 XOR (u[0] XOR ... XOR u[t-1]).
    The two solutions differ by x0 and are bitwise complements.
    """
    w0 = 0
    x = 0
    for t in range(p):
        if x:
            w0 |= 1 << t
        x ^= (u >> t) & 1
    w1 = w0 ^ ((1 << p) - 1)
    return w0, w1


# --------------------------------------------------------------------------
# trusted seeding from simulation
# --------------------------------------------------------------------------

def simulate_words(steps: int, diagonals: int, keep: int, period: int):
    """Minimal period and period-bit word of every settled diagonal.

    Same tail-recording simulation as period16_walk.py: only `keep` rows are
    materialised, so memory is O(keep * diagonals / 8), not O(T * D).
    """
    base = steps - keep + 1
    base -= base % period
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
            rows.append(np.frombuffer(win.to_bytes(nbytes, "little"),
                                      dtype=np.uint8))
        state = ((state << 1) ^ (state | (state >> 1))) & mask
    sim_s = time.perf_counter() - t0

    cands = [c for c in (1, 2, 4, 8, 16, 32, 64, 128) if len(rows) > 2 * c]
    period_of = np.zeros(diagonals, dtype=np.int32)
    for c in reversed(cands):          # large -> small so the minimum wins
        diff = np.zeros(nbytes, dtype=np.uint8)
        for i in range(len(rows) - c):
            diff |= rows[i] ^ rows[i + c]
        ok_c = np.unpackbits(~diff, bitorder="little", count=diagonals).astype(bool)
        period_of[ok_c] = c

    words = np.zeros(diagonals, dtype=np.uint64)
    for i in range(period):
        bits = np.unpackbits(rows[i], bitorder="little", count=diagonals)
        words |= bits.astype(np.uint64) << np.uint64(i)

    return period_of, words, sim_s, base


def pick_seed(period_of: np.ndarray, words: np.ndarray, period: int,
              min_run: int = 4096, margin: int = 1000) -> int:
    """A d inside the period-`period` regime with room to validate forward.

    Seeding early is wrong twice over: d < 2 indexes words[d-2] negatively, and
    the low-d transient is period 1/2/4/8, so the walk would spend its budget
    rediscovering the early doublings instead of starting where the period is
    already `period`. Start `margin` past the first period-`period` diagonal.
    """
    settled = (period_of > 0) & (period_of <= period)
    at_period = np.flatnonzero(period_of == period)
    if at_period.size == 0:
        raise RuntimeError(
            f"no diagonal of minimal period {period} in the simulated range; "
            f"simulate more diagonals (the 16->32 transition is at d=87867)")
    start = max(2, int(at_period[0]) + margin)
    n = len(settled)
    run = 0
    for d in range(start, n):
        run = run + 1 if settled[d] else 0
        if run >= min_run:
            return d - min_run + 1
    raise RuntimeError("no settled run long enough to seed the walk")


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------

def gate_lemma_a(period_of, words, lo, hi) -> dict:
    tested = bad = 0
    for d in range(lo + 2, hi):
        if period_of[d] == 0 or period_of[d - 1] == 0 or period_of[d - 2] == 0:
            continue
        tested += 1
        if (int(words[d]) == 0) != (int(words[d - 2]) == int(words[d - 1])):
            bad += 1
    return {"tested": tested, "violations": bad, "ok": bad == 0}


def gate_map_matches_simulation(words, d0, n_check, period) -> dict:
    out = np.zeros(n_check, dtype=np.uint64)
    done = _replay(np.uint64(words[d0 - 2]), np.uint64(words[d0 - 1]),
                   period, out)
    ref = words[d0:d0 + done]
    mismatches = int((out[:done] != ref).sum())
    first_bad = None
    if mismatches:
        first_bad = int(d0 + np.flatnonzero(out[:done] != ref)[0])
    return {"checked": int(done), "requested": int(n_check),
            "mismatches": mismatches, "first_mismatch_d": first_bad,
            "ok": mismatches == 0 and done > 0}


# --------------------------------------------------------------------------
# the walk
# --------------------------------------------------------------------------

def walk_for_doubling(u0: int, v0: int, d0: int, period: int, max_d: int,
                      max_branches: int, chunk: int = 1 << 30) -> dict:
    """Follow every branch until each one doubles or runs past max_d."""
    stack = [(int(u0), int(v0), int(d0), "root")]
    events, exhausted, branch_points = [], [], []
    steps_done = 0
    t0 = time.perf_counter()

    while stack:
        if len(events) + len(exhausted) + len(stack) > max_branches:
            raise RuntimeError(f"branch explosion past {max_branches}")
        u, v, d, label = stack.pop()
        while True:
            remaining = max_d - d
            if remaining <= 0:
                exhausted.append({"branch": label, "reached_d": d})
                break
            n = int(min(chunk, remaining))
            i, flag, u_at, v_at = _walk(np.uint64(u), np.uint64(v), period, n)
            steps_done += i
            d += i
            u, v = int(u_at), int(v_at)
            if flag == -1:
                continue                        # chunk boundary, keep going
            if flag == FLAG_DOUBLE:
                q = 1
                while q < period:
                    blk = u & ((1 << q) - 1)
                    if all(((u >> (j * q)) & ((1 << q) - 1)) == blk
                           for j in range(1, period // q)):
                        break
                    q *= 2
                events.append({
                    "branch": label, "d": d,
                    "collision_pair": [d - 2, d - 1],
                    "predecessor_word": f"0x{u:08x}",
                    "minimal_period_of_predecessor": q,
                    "popcount_at_minimal_period": bin(u & ((1 << q) - 1)).count("1"),
                    "period_before": period, "period_after": 2 * period,
                })
                break
            # FLAG_BRANCH: v == 0, even parity, two period-p continuations
            w0, w1 = branch_words(u, period)
            branch_points.append({"branch": label, "d": d,
                                  "predecessor_word": f"0x{u:08x}",
                                  "continuations": [f"0x{w0:08x}", f"0x{w1:08x}"]})
            stack.append((v, w0, d + 1, label + "0"))
            stack.append((v, w1, d + 1, label + "1"))
            break

    return {"events": events, "exhausted": exhausted,
            "branch_points": branch_points, "steps": steps_done,
            "elapsed_s": round(time.perf_counter() - t0, 2)}


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diagonals", type=int, default=200_000,
                    help="diagonals to simulate for the trusted seed")
    ap.add_argument("--keep", type=int, default=512,
                    help="tail rows recorded (memory is keep*diagonals/8)")
    ap.add_argument("--max-d", type=int, default=12_000_000_000,
                    help="stop each branch at this d")
    ap.add_argument("--max-branches", type=int, default=64)
    ap.add_argument("--check-bits", type=int, default=50_000,
                    help="diagonals of map-vs-simulation agreement to require")
    ap.add_argument("--out", default="data/wedge/pattern_map_walk32.json")
    ap.add_argument("--pretty", action="store_true")
    a = ap.parse_args(argv)

    if not NUMBA:
        raise SystemExit("numba is required: pip install numba")

    steps = int(a.diagonals * 1.3389 * 1.10) + 4096
    if a.pretty:
        print(f"simulating T={steps:,} for D={a.diagonals:,} "
              f"(tail {a.keep} rows)...", flush=True)
    period_of, words, sim_s, base = simulate_words(steps, a.diagonals,
                                                   a.keep, PERIOD)
    hist = {int(c): int((period_of == c).sum())
            for c in np.unique(period_of) if c > 0}
    unsettled = int((period_of == 0).sum())

    over16 = np.flatnonzero(period_of == PERIOD)
    first_p32 = int(over16[0]) if over16.size else None

    d0 = pick_seed(period_of, words, PERIOD)
    hi = min(a.diagonals, d0 + a.check_bits)

    g_a = gate_lemma_a(period_of, words, d0 - 2, hi)
    g_map = gate_map_matches_simulation(words, d0, hi - d0, PERIOD)
    g_seed = {"d0": d0,
              "period_at_d0_minus_2": int(period_of[d0 - 2]),
              "period_at_d0_minus_1": int(period_of[d0 - 1]),
              "ok": bool(period_of[d0 - 1] == PERIOD or period_of[d0 - 2] == PERIOD)}

    gates_ok = g_a["ok"] and g_map["ok"] and g_seed["ok"]
    if a.pretty:
        print(f"  simulation {sim_s:.1f}s   period histogram {hist}"
              f"   unsettled {unsettled:,}")
        print(f"  first period-32 diagonal: {first_p32}")
        print(f"  seed d0={d0:,}  u=0x{int(words[d0-2]):08x}  "
              f"v=0x{int(words[d0-1]):08x}")
        print(f"  gate lemma-A          : {g_a['tested']:,} tested, "
              f"{g_a['violations']} violations")
        print(f"  gate map==simulation  : {g_map['checked']:,} diagonals, "
              f"{g_map['mismatches']} mismatches")
        print(f"  gate seed is period-32: {g_seed['ok']}")

    result = None
    if gates_ok:
        if a.pretty:
            print(f"\nwalking to d={a.max_d:,} ...", flush=True)
        result = walk_for_doubling(int(words[d0 - 2]), int(words[d0 - 1]),
                                   d0, PERIOD, a.max_d, a.max_branches)
        if a.pretty:
            rate = result["steps"] / max(result["elapsed_s"], 1e-9) / 1e6
            print(f"  {result['steps']:,} map steps in "
                  f"{result['elapsed_s']}s ({rate:.1f} M/s)")
            print(f"  branch points: {len(result['branch_points'])}")
            for e in result["events"]:
                print(f"  DOUBLING  branch={e['branch']}  d={e['d']:,}  "
                      f"pred={e['predecessor_word']} "
                      f"minperiod={e['minimal_period_of_predecessor']} "
                      f"-> period {e['period_after']}")
            for x in result["exhausted"]:
                print(f"  no doubling on branch={x['branch']} "
                      f"by d={x['reached_d']:,}")

    payload = {
        "experiment": "pattern_map_walk32",
        "period": PERIOD,
        "simulation": {"steps": steps, "diagonals": a.diagonals,
                       "keep": a.keep, "base_row": base,
                       "elapsed_s": round(sim_s, 2),
                       "period_histogram": hist, "unsettled": unsettled,
                       "first_period32_diagonal": first_p32},
        "seed": {"d0": d0, "u": f"0x{int(words[d0-2]):08x}",
                 "v": f"0x{int(words[d0-1]):08x}"},
        "gates": {"lemma_a": g_a, "map_vs_simulation": g_map, "seed": g_seed,
                  "all_ok": gates_ok},
        "walk": result,
        "max_d": a.max_d,
    }
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    if a.pretty:
        print(f"\nwrote {a.out}")

    if not gates_ok:
        print("GATES FAILED - walk not attempted", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

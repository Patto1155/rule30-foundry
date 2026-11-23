#!/usr/bin/env python
"""Exact Myhill-Nerode LOWER BOUND on the base-2 DFAO state count s*(n).

Why this exists
---------------
The certified ``s*(n)`` curve stops at n=64 because each point is a SAT
minimisation with don't-cares, and the deciding refutation at n=64 was
already a 2.26 GB proof. But a *lower* bound on ``s*(n)`` does not need SAT.
It is the textbook distinguishability argument, it is exact, and it costs
one sort of the prefix.

The bound (purpose: exact-exclusion)
------------------------------------
Convention is the repo's (`prize_lab.digits_for`): index ``i >= 0`` is
written in base 2 with no leading zeros (``0`` is the single digit ``0``).

For any ``J >= 0`` and ``L`` with ``2^J * L <= n``, every index
``i = 2^J * m + r`` with ``1 <= m < L``, ``0 <= r < 2^J`` lies in the prefix,
and because ``m >= 1`` its digit string is ``digits(m)`` followed by the
``J``-digit zero-padded form of ``r``.

* **MSD-first.** The automaton reads ``digits(m)`` first, reaching a state
  ``q(m)``, and the output at ``i`` is then a function of ``q(m)`` and the
  ``j`` padded digits alone. So the vector
  ``col(m) = ( c[2^j m + r] : 0 <= j <= J, 0 <= r < 2^j )`` is a function of
  ``q(m)``. Two ``m`` with different vectors are in different states:
  ``s*_msd(n) >= #distinct col(m), 1 <= m < L``.
* **LSD-first.** The automaton reads the ``J`` low digits of ``r`` first,
  reaching ``p(r)``, and the output is then a function of ``p(r)`` and ``m``.
  So ``row(r) = ( c[2^J m + r] : 1 <= m < L )`` is a function of ``p(r)``:
  ``s*_lsd(n) >= #distinct row(r), 0 <= r < 2^J``.

Every coordinate compared is a defined prefix bit, so distinct vectors are a
clique of pairwise-incompatible states - no don't-care can merge them. The
bound is maximised over ``J``. It is a lower bound only: the true minimum
also has to separate indices this family does not cover.

What it is NOT
--------------
Not evidence of complexity. ``counting_bound.py`` semantics apply: a uniform
random string of the same length gets the same bound (the 7-seed band is
reported alongside, and a matching control is the *expected* outcome here,
not a reassurance). It is a bounded exact exclusion - "no base-2 DFAO with
fewer than LB states generates the first n centre bits" - and it replaces
"n=72+ needs a better encoding" as the practical frontier for lower bounds.

Usage
-----
    python experiments/dfao_state_lower_bound.py --self-test
    python experiments/dfao_state_lower_bound.py --out data/prize/<date>-dfao-state-lower-bound.json
    python experiments/dfao_state_lower_bound.py --verify data/prize/<date>-dfao-state-lower-bound.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_TYPE = "rule30.dfao_state_lower_bound"
ARTIFACT_VERSION = 1
DEFAULT_NS = [8, 16, 24, 32, 40, 48, 56, 64, 128, 256, 1024, 4096, 16384,
              65536, 262144, 1_000_000, 4_000_000, 10_000_000]
# Certified exact minima to cross-check against (lower bound must not exceed).
CERTIFIED = {("msd", 56): 13, ("msd", 64): 15}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_bits(path: Path, n_bits: int) -> np.ndarray:
    """Canonical centre-column dumps are LSB-first. Never omit bitorder."""
    raw = np.fromfile(str(path), dtype=np.uint8, count=(n_bits + 7) // 8)
    bits = np.unpackbits(raw, bitorder="little")[:n_bits]
    if bits.size < n_bits:
        raise SystemExit(f"{path} holds {bits.size:,} bits, need {n_bits:,}")
    return bits


def naive_center(n: int) -> list[int]:
    """Independent single-seed Rule 30 centre column (big-int rows)."""
    row, out = 1, []
    for t in range(n):
        out.append((row >> t) & 1)
        row = (row << 2) ^ ((row << 1) | row)   # left ^ (centre | right)
    return out


def _count_distinct_rows(mat: np.ndarray) -> int:
    packed = np.packbits(mat, axis=1, bitorder="big")   # row key only
    view = np.ascontiguousarray(packed).view(
        np.dtype((np.void, packed.shape[1])))
    return int(np.unique(view).size)


def msd_bound(c: np.ndarray, J: int) -> int:
    n = c.size
    L = n >> J
    if L < 2:
        return 0
    blocks = [c[(1 << j):(L << j)].reshape(L - 1, 1 << j) for j in range(J + 1)]
    return _count_distinct_rows(np.concatenate(blocks, axis=1))


def lsd_bound(c: np.ndarray, J: int) -> int:
    n = c.size
    L = n >> J
    if L < 2:
        return 1 if n else 0
    mat = c[(1 << J):(L << J)].reshape(L - 1, 1 << J).T
    return _count_distinct_rows(mat)


def lower_bounds(c: np.ndarray) -> dict:
    n = c.size
    best = {"msd": (1, 0), "lsd": (1, 0)}
    for J in range(0, max(1, n.bit_length())):
        if (n >> J) < 2:
            break
        for name, fn in (("msd", msd_bound), ("lsd", lsd_bound)):
            v = fn(c, J)
            if v > best[name][0]:
                best[name] = (v, J)
    return {"n": n,
            "msd_lower_bound": best["msd"][0], "msd_J": best["msd"][1],
            "lsd_lower_bound": best["lsd"][0], "lsd_J": best["lsd"][1]}


def log2_class(s: int) -> float:
    """log2 of the number of s-state binary DFAO tables, s^(2s) * 2^s."""
    return 2 * s * math.log2(s) + s if s > 0 else 0.0


# ----------------------------------------------------------------- self-test
def run_dfao(delta, out, i: int, direction: str) -> int:
    digits = [int(ch) for ch in bin(i)[2:]]
    if direction == "lsd":
        digits.reverse()
    q = 0
    for d in digits:
        q = delta[q][d]
    return out[q]


def self_test() -> dict:
    checks: list[dict] = []

    def record(name, ok, detail):
        checks.append({"check": name, "passed": bool(ok), "detail": str(detail)})
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")

    # 1. Soundness: for a sequence GENERATED by an s-state DFAO the bound can
    #    never exceed s. 400 random automata, both directions, several n.
    rng = np.random.default_rng(20261123)
    worst = 0
    bad = 0
    tight = 0
    trials = 0
    for _ in range(200):
        s = int(rng.integers(1, 13))
        delta = rng.integers(0, s, size=(s, 2)).tolist()
        out = rng.integers(0, 2, size=s).tolist()
        for direction in ("msd", "lsd"):
            for n in (64, 1000, 4096):
                seq = np.array([run_dfao(delta, out, i, direction)
                                for i in range(n)], dtype=np.uint8)
                lb = lower_bounds(seq)[f"{direction}_lower_bound"]
                trials += 1
                bad += lb > s
                tight += lb == s
                worst = max(worst, lb - s)
    record("bound never exceeds the generating automaton's state count",
           bad == 0, f"{trials} cases, {bad} violations, {tight} tight")

    # 2. Positive control: Thue-Morse is a 2-state DFAO in both directions.
    tm = np.array([bin(i).count("1") & 1 for i in range(1 << 16)],
                  dtype=np.uint8)
    r = lower_bounds(tm)
    record("Thue-Morse bound is exactly 2 at n=65536, both directions",
           r["msd_lower_bound"] == 2 and r["lsd_lower_bound"] == 2,
           f"msd {r['msd_lower_bound']}, lsd {r['lsd_lower_bound']}")

    # 3. Cross-check against the SAT-certified exact minima for the centre
    #    column: a lower bound above a certified s* would refute one of them.
    center = np.array(naive_center(64), dtype=np.uint8)
    for (direction, n), s_star in sorted(CERTIFIED.items()):
        lb = lower_bounds(center[:n])[f"{direction}_lower_bound"]
        record(f"bound <= certified s*_{direction}({n}) = {s_star}",
               lb <= s_star, f"lower bound {lb}")

    # 4. A random string's bound must grow (the instrument is not capped).
    rnd = np.random.default_rng(7).integers(0, 2, size=1 << 16, dtype=np.uint8)
    r = lower_bounds(rnd)
    record("random 65536-bit string exceeds 1000 states, both directions",
           min(r["msd_lower_bound"], r["lsd_lower_bound"]) > 1000,
           f"msd {r['msd_lower_bound']}, lsd {r['lsd_lower_bound']}")

    passed = all(c["passed"] for c in checks)
    log(f"self-test {'PASSED' if passed else 'FAILED'}")
    return {"passed": passed, "checks": checks}


# ---------------------------------------------------------------------- main
def curve(bits: np.ndarray, ns: list[int]) -> list[dict]:
    rows = []
    for n in ns:
        r = lower_bounds(bits[:n])
        for d in ("msd", "lsd"):
            s = r[f"{d}_lower_bound"]
            r[f"{d}_log2_class_at_bound"] = round(log2_class(s), 1)
        rows.append(r)
    return rows


def build(bitstream: Path, ns: list[int], seeds: list[int]) -> dict:
    n_max = max(ns)
    bits = load_bits(bitstream, n_max)
    naive = np.array(naive_center(4096), dtype=np.uint8)
    if not np.array_equal(bits[:4096], naive):
        raise SystemExit("bitstream does not match the naive centre column "
                         "(check bit order)")
    center = curve(bits, ns)
    controls = {}
    for seed in seeds:
        ctrl = np.random.default_rng(seed).integers(
            0, 2, size=n_max, dtype=np.uint8)
        controls[str(seed)] = curve(ctrl, ns)
        log(f"  control seed {seed} done")
    table = []
    for k, row in enumerate(center):
        entry = dict(row)
        for d in ("msd", "lsd"):
            band = sorted(controls[s][k][f"{d}_lower_bound"] for s in controls)
            entry[f"{d}_random_band"] = [band[0], band[-1]] if band else None
            v = row[f"{d}_lower_bound"]
            entry[f"{d}_vs_band"] = (
                None if not band else
                "below" if v < band[0] else "above" if v > band[-1]
                else "inside")
        table.append(entry)
    return {
        "source": {"bitstream": bitstream.name, "bits": n_max,
                   "bitorder": "little",
                   "sha256_prefix_bits": hashlib.sha256(
                       np.packbits(bits, bitorder="little").tobytes()
                   ).hexdigest(),
                   "naive_prefix_checked_bits": 4096},
        "control_seeds": seeds,
        "table": table,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bitstream", type=Path,
                    default=ROOT / "data" / "center_col_10M.bin")
    ap.add_argument("--ns", default=",".join(map(str, DEFAULT_NS)))
    ap.add_argument("--seeds", default=",".join(str(30 + i) for i in range(7)))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--verify", type=Path, default=None)
    args = ap.parse_args()

    log("self-test:")
    st = self_test()
    if not st["passed"]:
        return 1
    if args.self_test:
        return 0

    if args.verify:
        art = json.loads(args.verify.read_text(encoding="utf-8"))
        ns = [r["n"] for r in art["table"]]
        fresh = build(args.bitstream, ns, art["control_seeds"])
        ok = (fresh["table"] == art["table"]
              and fresh["source"] == art["source"])
        log("VERIFY " + ("PASSED" if ok else "FAILED") +
            f": {len(ns)} rows recomputed")
        return 0 if ok else 1

    if not args.bitstream.exists():
        raise SystemExit(f"bitstream not found: {args.bitstream} "
                         "(python tools/convert_golden_bitstream.py)")
    started = time.time()
    ns = [int(x) for x in args.ns.split(",")]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    body = build(args.bitstream, ns, seeds)
    artifact = {
        "artifact_type": ARTIFACT_TYPE,
        "artifact_version": ARTIFACT_VERSION,
        "claim_level": "Certificate (bounded exact exclusion; lower bound only)",
        "purpose": "exact-exclusion",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - started, 2),
        "statement": (
            "No base-2 DFAO (repo digit convention, no leading zeros) with "
            "fewer than the listed number of states generates the first n "
            "single-seed centre-column bits. Myhill-Nerode distinguishability "
            "over fully defined prefix positions; lower bound on s*(n) only."),
        "not_evidence_of_complexity": (
            "A uniform random string gets the same bound (random_band). The "
            "class at the bound is far smaller than 2^n only relative to "
            "labelled tables; see log2_class_at_bound vs n."),
        "self_test": st,
        **body,
    }
    text = json.dumps(artifact, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="")
        log(f"wrote {args.out}")
    else:
        print(text)
    for r in artifact["table"]:
        log(f"n={r['n']:>10,}  msd >= {r['msd_lower_bound']:>7,} "
            f"{r['msd_random_band']} {r['msd_vs_band']:<7} "
            f"lsd >= {r['lsd_lower_bound']:>7,} {r['lsd_random_band']} "
            f"{r['lsd_vs_band']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

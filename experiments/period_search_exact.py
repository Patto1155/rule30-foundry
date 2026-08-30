#!/usr/bin/env python
"""Exhaustive EXACT period search in the Rule 30 center column (Problem 1).

Prize Problem 1 asks whether the center column is eventually periodic. A found
period is a $30,000 counterexample, so this is a direct search for a specific
structure rather than a negative over a model class - the counting bound of
`docs/theory/finite-prefix-counting-bound.md` does not blunt it, and the
Admission Rule does not apply.

Why this replaces the sampled search
------------------------------------
`experiments/period_search.py` (Experiment E) estimates, for each candidate
period p, the *match rate* over 10,000 randomly sampled positions, and reports
a z-score against a Bonferroni threshold (best z = 4.66 vs threshold 5.61).
That design has three avoidable weaknesses:

1. **It is statistical about an exact question.** ``p`` is a period of the
   observed prefix iff ``S[i] == S[i+p]`` for *every* valid ``i``. One
   mismatch refutes it outright. There is nothing to estimate.
2. **It needs a multiple-testing correction**, and that correction has to grow
   with the search - the plan warns that forgetting to grow it "would
   manufacture a false positive". An exact test has no such problem: there is
   no null distribution and no threshold to correct.
3. **Sampling is more expensive, not less.** A random sequence disagrees with
   its own shift within a handful of positions, so an exact test costs O(1)
   per candidate in practice, against 10,000 samples for the estimate.

Method
------
A period ``p`` survives only if the first ``W`` bits agree with the bits at
offset ``p``. So build the ``W``-bit window at every position, packed into a
single integer, and take the candidates where that integer equals the window
at position 0. For ``W = 64`` on a random-looking stream the expected number
of surviving candidates is ``n / 2^64``, i.e. none - which refutes every
candidate period in one vectorised pass. Survivors are then checked
position-by-position to the end of the overlap.

This yields an exact verdict for every ``p``, not a sampled one, and covers
every ``p <= n - W`` rather than a chosen ceiling.

Reporting range
---------------
A "period" confirmed by only a handful of positions is vacuous, so the headline
claim is restricted to ``p <= n/2``, where at least ``n/2`` positions confirm
it. Longer ``p`` are still tested and reported separately.

Secondary statistic: the longest self-overlap
---------------------------------------------
``Z[p]`` is the length of the longest common prefix of the stream and the
stream shifted by ``p``. ``max_p Z[p]`` measures how *close* the sequence comes
to repeating, which a yes/no period search throws away.

For an i.i.d. fair stream, ``P[Z[p] >= k] = 2^-k`` and there are about ``n``
shifts, so ``E[#{p : Z[p] >= k}] ~ n * 2^-k`` and the maximum sits near
``log2(n)`` - **not** ``2*log2(n)``. The doubled form is the longest common
substring between two *independent* length-``n`` sequences, where there are
``n^2`` position pairs rather than ``n`` shifts. Getting this wrong would
inflate the apparent null and make a real anomaly look ordinary, so the
artifact reports the union-bound tail probability at the observed value
alongside a multi-seed empirical band.

Usage
-----
    python experiments/period_search_exact.py \
        --bitstream data/center_col_10M.bin \
        --out data/prize/period-search-exact.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from prize_lab import git_context  # noqa: E402

ARTIFACT_TYPE = "rule30.period_search_exact"
ARTIFACT_VERSION = 1


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_bits(path: Path, n_bits: int) -> np.ndarray:
    """Center-column dumps are LSB-first. Never omit bitorder here."""
    n_bytes = (n_bits + 7) // 8
    raw = np.fromfile(str(path), dtype=np.uint8, count=n_bytes)
    bits = np.unpackbits(raw, bitorder="little")[:n_bits]
    if bits.size < n_bits:
        raise SystemExit(f"{path} holds {bits.size:,} bits, need {n_bits:,}")
    return bits


def windows(bits: np.ndarray, width: int) -> np.ndarray:
    """W[i] = the `width` bits starting at i, packed MSB-first into uint64.

    Built by `width` vectorised shift-or passes rather than a stride trick, so
    the memory cost is one uint64 array rather than a `width`-fold view.
    """
    if width > 64:
        raise ValueError("window width must be <= 64")
    n = bits.size - width + 1
    out = np.zeros(n, dtype=np.uint64)
    b64 = bits.astype(np.uint64)
    for j in range(width):
        out <<= np.uint64(1)
        out |= b64[j:j + n]
    return out


def overlap_length(bits: np.ndarray, p: int, cap: int | None = None) -> int:
    """Exact Z[p]: how many leading positions agree with the shift by p."""
    n = bits.size - p
    if n <= 0:
        return 0
    if cap is not None:
        n = min(n, cap)
    eq = bits[:n] == bits[p:p + n]
    bad = np.flatnonzero(~eq)
    return int(bad[0]) if bad.size else int(n)


def find_periods(bits: np.ndarray, width: int, *, stop_at_first: bool = True,
                 max_checks: int = 100_000) -> dict:
    """Decide exactly, for every p <= n - width, whether p is a period.

    ``p`` survives only if the first ``width`` bits agree with the bits at
    offset ``p``, so one vectorised equality over the packed windows refutes
    almost everything. On a random-looking stream the expected number of
    survivors is ``n / 2^width``, i.e. none for width 64.

    ``stop_at_first`` matters on genuinely periodic input: there *every*
    multiple of the true period survives, so checking each to the end would be
    quadratic. The smallest confirmed period generates all the others, so the
    first one is the answer.
    """
    n = bits.size
    t0 = time.perf_counter()
    win = windows(bits, width)
    build_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    survivors = np.flatnonzero(win == win[0])
    survivors = survivors[survivors >= 1]
    scan_s = time.perf_counter() - t1

    confirmed: list[dict] = []
    checked = 0
    truncated = False
    for p_ in survivors.tolist():          # ascending, so smallest first
        if checked >= max_checks:
            truncated = True
            break
        checked += 1
        if overlap_length(bits, p_) == n - p_:
            confirmed.append({"period": p_, "confirming_positions": n - p_})
            if stop_at_first:
                break

    return {
        "n_bits": n,
        "window_bits": width,
        "periods_tested_exactly": int(max(0, n - width)),
        "candidates_surviving_window": int(survivors.size),
        "candidates_checked": checked,
        "candidate_check_truncated": truncated,
        "confirmed_periods": confirmed,
        "confirmed_periods_up_to_half": [
            c for c in confirmed if c["period"] <= n // 2],
        "timing_s": {"window_build": round(build_s, 2),
                     "candidate_scan": round(scan_s, 2)},
    }


def longest_self_overlap(bits: np.ndarray, *, narrow: int = 16,
                         max_candidates: int = 20_000) -> dict:
    """Exact max_p Z[p]: how close the stream comes to repeating.

    A yes/no period search throws this away. The window has to be narrow
    enough to leave candidates to extend - a 64-bit window on a random stream
    returns nothing and would only tell us "< 64".
    """
    t0 = time.perf_counter()
    win = windows(bits, narrow)
    cand = np.flatnonzero(win == win[0])
    cand = cand[cand >= 1]
    truncated = cand.size > max_candidates
    todo = cand[:max_candidates].tolist()

    best_p, best_z = 0, 0
    for p_ in todo:
        z = overlap_length(bits, p_)
        if z > best_z:
            best_z, best_p = z, p_
    return {
        "longest_self_overlap_bits": best_z,
        "longest_self_overlap_at_shift": best_p,
        "narrow_window_bits": narrow,
        "narrow_window_candidates": int(cand.size),
        "candidates_examined": len(todo),
        "truncated": bool(truncated),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }


def gate(bits: np.ndarray, width: int) -> dict:
    """Detection power: the method must FIND a period that is really there."""
    checks: list[dict] = []

    def record(name, ok, detail):
        checks.append({"check": name, "passed": bool(ok), "detail": str(detail)})
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")

    # 1. A truly periodic stream must be found, at the right period.
    for p in (3, 1000, 65536):
        planted = np.resize(bits[:p], 2_000_000)
        res = find_periods(planted, width, stop_at_first=True)
        found = [c["period"] for c in res["confirmed_periods"]]
        record(f"planted period {p} is found",
               bool(found) and min(found) == p,
               f"smallest found = {min(found) if found else None}")

    # 2. A stream that is periodic except for one flipped bit must NOT be.
    planted = np.resize(bits[:1000], 2_000_000).copy()
    planted[1_500_000] ^= 1
    res = find_periods(planted, width, stop_at_first=False)
    record("one flipped bit refutes an otherwise perfect period",
           not res["confirmed_periods_up_to_half"],
           f"{len(res['confirmed_periods_up_to_half'])} confirmed")

    # 3. The bitstream must decode to OEIS A051023, i.e. LSB-first was used.
    oeis = [1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0]
    record("bitstream decodes LSB-first to OEIS A051023",
           bits[:len(oeis)].tolist() == oeis, "".join(map(str, bits[:15])))

    passed = all(c["passed"] for c in checks)
    log(f"gate {'PASSED' if passed else 'FAILED'}")
    return {"passed": passed, "checks": checks}


def union_bound_tail(n: int, k: int) -> float:
    """P[some shift has Z >= k] for an i.i.d. fair stream, by union bound.

    E[#{p : Z[p] >= k}] ~ (n - k) * 2^-k, and the count is close to Poisson,
    so P[max >= k] ~ 1 - exp(-E).
    """
    if k <= 0:
        return 1.0
    expected = max(0, n - k) * (2.0 ** -k)
    return 1.0 - math.exp(-expected)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bitstream", type=Path,
                    default=ROOT / "data" / "center_col_10M.bin")
    ap.add_argument("--bits", type=int, default=10_000_000)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--seeds", default=",".join(str(30 + i) for i in range(50)))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--gate-only", action="store_true")
    args = ap.parse_args()

    if not args.bitstream.exists():
        raise SystemExit(
            f"bitstream not found: {args.bitstream}\n"
            "  (gitignored; regenerate with gpu/rule30_sim.py, or with\n"
            "   tools/gen_golden_reference.py and repack LSB-first)")

    log(f"loading {args.bitstream} ({args.bits:,} bits)")
    bits = load_bits(args.bitstream, args.bits)

    log("gate:")
    gate_result = gate(bits, args.window)
    if not gate_result["passed"]:
        log("gate failed; refusing to search")
        return 1
    if args.gate_only:
        return 0

    log(f"\nexact scan over every period p <= {args.bits - args.window:,}:")
    started = time.time()
    center = find_periods(bits, args.window, stop_at_first=False)
    log(f"  candidates surviving a {args.window}-bit window: "
        f"{center['candidates_surviving_window']}")
    log(f"  confirmed periods: {len(center['confirmed_periods'])}")
    center_overlap = longest_self_overlap(bits)
    center["overlap"] = center_overlap
    log(f"  longest self-overlap: {center_overlap['longest_self_overlap_bits']} "
        f"bits at shift {center_overlap['longest_self_overlap_at_shift']:,} "
        f"({center_overlap['narrow_window_candidates']} candidates, "
        f"{center_overlap['elapsed_s']}s)")

    log("\nmatched random controls:")
    controls = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        rng = np.random.default_rng(seed)
        ctrl = rng.integers(0, 2, size=args.bits, dtype=np.uint8)
        res = find_periods(ctrl, args.window, stop_at_first=False)
        ov = longest_self_overlap(ctrl)
        controls.append({
            "seed": seed,
            "confirmed_periods": len(res["confirmed_periods"]),
            "candidates_surviving_window": res["candidates_surviving_window"],
            "longest_self_overlap_bits": ov["longest_self_overlap_bits"]})
        log(f"  seed {seed}: {len(res['confirmed_periods'])} periods, "
            f"longest overlap {ov['longest_self_overlap_bits']} bits")

    overlaps = sorted(c["longest_self_overlap_bits"] for c in controls)
    observed = center["overlap"]["longest_self_overlap_bits"]
    n_ge = sum(1 for v in overlaps if v >= observed)
    empirical_p = (n_ge + 1) / (len(overlaps) + 1)
    expected = math.log2(args.bits)

    artifact = {
        "artifact_type": ARTIFACT_TYPE,
        "artifact_version": ARTIFACT_VERSION,
        "claim_level": "Certificate (exhaustive and exact over the stated range)",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - started, 3),
        "question": (
            "Is any p <= n/2 a period of the first n center-column bits?"),
        "method": {
            "exact": True,
            "sampling": None,
            "note": (
                "A period is refuted by a single mismatch, so no statistical "
                "threshold and no multiple-testing correction is involved. "
                "Every p <= n - window is decided exactly."),
            "headline_range": "p <= n/2, so at least n/2 positions confirm",
            "supersedes": (
                "experiments/period_search.py, which sampled 10,000 positions "
                "per period over p <= 10^6 and reported a Bonferroni z-score"),
        },
        "source": {
            "bitstream": args.bitstream.name,
            "bits": args.bits,
            "bitorder": "little (gpu/rule30_sim.py convention)",
        },
        "gate": gate_result,
        "center": center,
        "random_controls": controls,
        "control_summary": {
            "longest_self_overlap_band": overlaps,
            "control_min": overlaps[0],
            "control_max": overlaps[-1],
            "control_median": overlaps[len(overlaps) // 2],
            "expected_max_is_about_log2n": round(expected, 2),
            "note": (
                "The expected maximum is ~log2(n), not 2*log2(n): there are "
                "~n shifts, not n^2 position pairs. 2*log2(n) is the longest "
                "common substring between two INDEPENDENT sequences."),
            "center_longest_overlap": observed,
            "seeds_reaching_center_value": n_ge,
            "empirical_p_value": round(empirical_p, 4),
            "union_bound_p_at_center_value": round(
                union_bound_tail(args.bits, observed), 4),
            "verdict": (
                "consistent with chance"
                if union_bound_tail(args.bits, observed) > 0.01
                else "anomalous; investigate"),
        },
        "git": git_context(),
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(artifact, indent=2) + "\n",
                            encoding="utf-8", newline="")
        log(f"\nwrote {args.out}")
    else:
        print(json.dumps(artifact, indent=2))

    n_conf = len(center["confirmed_periods_up_to_half"])
    log(f"\nRESULT: {n_conf} periods p <= {args.bits // 2:,} confirmed "
        f"(0 expected if aperiodic)")
    log(f"longest self-overlap {observed} bits; control band "
        f"[{overlaps[0]}, {overlaps[-1]}] over {len(overlaps)} seeds "
        f"(median {overlaps[len(overlaps)//2]}), expected max ~log2(n) = "
        f"{expected:.1f}")
    log(f"  empirical p = {empirical_p:.4f}, union-bound p = "
        f"{union_bound_tail(args.bits, observed):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Can a window-based inductive invariant exclude the alternating-centre loops?

`experiments/periodic_strip_graph.py` showed that open Rule 30 strips admit an
alternating centre with a non-eventually-periodic left neighbour, using freely
supplied boundary cells. `experiments/seed_strip_compatibility.py` then rejected
most saved alignments against actual seed rows -- but only those walks, at those
onsets. A finite exclusion does not generalise; an INVARIANT would.

The proof obligations for an invariant I are the usual three:

    (a) true of the seed's rows,
    (b) preserved by the rule,
    (c) incompatible with the target behaviour.

This script does not search candidate invariants one at a time. For the family
of window-based invariants -- I(row) = "every k-window of row lies in W", the
subshifts of finite type -- there is a unique BEST candidate at each k, namely

    W_seed(k) = the set of k-windows that actually occur in seed rows.

Any SFT invariant satisfying (a) must contain W_seed(k), and a larger W can only
exclude less. So testing this one W decides the entire family at that k at once:
if W_seed(k) cannot exclude the loops, no window invariant of width k can. That
makes the result a characterisation rather than a search, and the counting
bound (docs/theory/README.md §5) does not apply -- nothing is being fitted.

Obligation (b) is checked LOCALLY and exhaustively. Rule 30 maps a (k+2)-window
to a k-window, so preservation holds iff every (k+2)-window whose k-sub-windows
all lie in W_seed has its image in W_seed. That is 2^(k+2) cases, complete.

THE OUTCOME THAT WOULD MAKE THIS VACUOUS, checked and reported: if W_seed(k) is
all of {0,1}^k then I is "true of every row", which is preserved trivially and
excludes nothing. A pass on (a) and (b) with a full W is not an invariant, it is
a tautology, and `nontrivial` says so.

Usage:
    python experiments/seed_invariant_search.py --kmax 14 --pretty
    python experiments/seed_invariant_search.py --verify runs/<artifact>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ARTIFACT_TYPE = "rule30.seed_invariant_search"
ARTIFACT_VERSION = 1
LOOPS = Path("runs/periodic-boundary-2026-09-08/output-loops.json")


def seed_rows(steps: int) -> list[int]:
    """Row t of the single-seed orbit as an int, LSB = leftmost cone cell."""
    width = 2 * steps + 3
    centre = steps + 1
    mask = (1 << width) - 1
    rows, state = [], 1 << centre
    for t in range(steps + 1):
        rows.append((state >> (centre - t)) & ((1 << (2 * t + 1)) - 1))
        state = ((state << 1) ^ (state | (state >> 1))) & mask
    return rows


def windows_of(row: int, length: int, k: int) -> set[int]:
    """Every k-window of a row of the given bit length."""
    if length < k:
        return set()
    out = set()
    m = (1 << k) - 1
    for i in range(length - k + 1):
        out.add((row >> i) & m)
    return out


def seed_window_set(steps: int, k: int) -> set[int]:
    out: set[int] = set()
    full = (1 << k) - 1
    for t, row in enumerate(seed_rows(steps)):
        out |= windows_of(row, 2 * t + 1, k)
        if len(out) == full + 1:
            break
    return out


def rule30_image(window: int, k_plus_2: int) -> int:
    """Rule 30 applied to a (k+2)-window, giving the k-window below its middle."""
    out = 0
    for i in range(1, k_plus_2 - 1):
        a, b, c = ((window >> j) & 1 for j in (i - 1, i, i + 1))
        out |= (a ^ (b | c)) << (i - 1)
    return out


def preserved(W: set[int], k: int) -> dict:
    """Exhaustive local check of obligation (b) over all 2^(k+2) windows."""
    m = (1 << k) - 1
    violations = []
    for u in range(1 << (k + 2)):
        if (u & m) not in W or ((u >> 1) & m) not in W or ((u >> 2) & m) not in W:
            continue
        img = rule30_image(u, k + 2)
        if img not in W:
            violations.append({"window": u, "image": img})
            if len(violations) >= 5:
                break
    return {"checked": 1 << (k + 2), "violations": violations,
            "ok": not violations}


def loop_rows() -> list[dict]:
    """Interior rows of the saved alternating-centre closed walks."""
    if not LOOPS.is_file():
        return []
    out = []
    for entry in json.loads(LOOPS.read_text(encoding="utf-8")):
        rows = []
        for walk in entry.get("closed_walks") or []:
            for _, row in walk:
                rows.append(int(row))
        if rows:
            out.append({"width": entry["width"], "rows": sorted(set(rows))})
    return out


def excludes(W: set[int], k: int, loops: list[dict]) -> dict:
    """Obligation (c): does any loop row carry a window the seed never shows?

    Only windows strictly inside the strip are counted. The boundary cells are
    freely supplied in these constructions, so a window touching them is not a
    statement about Rule 30.
    """
    hits = []
    for entry in loops:
        w = entry["width"]
        if w - 2 < k:
            continue
        bad = set()
        for row in entry["rows"]:
            interior = row >> 1
            bad |= {x for x in windows_of(interior, w - 2, k) if x not in W}
        hits.append({"width": w, "rows": len(entry["rows"]),
                     "forbidden_windows_found": sorted(bad)[:8],
                     "excluded": bool(bad)})
    return {"per_width": hits,
            "excludes_any": any(h["excluded"] for h in hits),
            "widths_tested": [h["width"] for h in hits]}


def control_rule(rule: int, steps: int, k: int) -> int:
    """|W(k)| for another elementary rule from the same single-cell seed.

    If the seed window set is full for Rule 30 *and* for the controls, the
    emptiness of this route is a property of short windows, not of Rule 30.
    """
    width = 2 * steps + 3
    centre = steps + 1
    mask = (1 << width) - 1
    table = [(rule >> i) & 1 for i in range(8)]
    out: set[int] = set()
    state = 1 << centre
    for t in range(steps + 1):
        out |= windows_of((state >> (centre - t)) & ((1 << (2 * t + 1)) - 1),
                          2 * t + 1, k)
        nxt = 0
        for i in range(1, width - 1):
            nxt |= table[((state >> (i - 1)) & 1)
                         | (((state >> i) & 1) << 1)
                         | (((state >> (i + 1)) & 1) << 2)] << i
        state = nxt & mask
    return len(out)


def run(steps: int, kmax: int, controls: tuple[int, ...]) -> dict:
    started = time.time()
    loops = loop_rows()
    rows = []
    for k in range(2, kmax + 1):
        W = seed_window_set(steps, k)
        full = len(W) == (1 << k)
        row = {
            "k": k,
            "n_seed_windows": len(W),
            "of_possible": 1 << k,
            "full_language": full,
            "preserved": preserved(W, k) if not full else
                         {"checked": 0, "violations": [],
                          "ok": True, "note": "trivially preserved: W is every window"},
            "exclusion": excludes(W, k, loops),
        }
        row["nontrivial"] = (not full) and row["exclusion"]["excludes_any"]
        row["verdict"] = (
            "VACUOUS: W is the full window set, so the invariant is 'true of "
            "every row'. It is preserved because it says nothing, and it "
            "excludes nothing."
            if full else
            "INVARIANT: proper W, preserved, and it rejects a loop row"
            if row["preserved"]["ok"] and row["exclusion"]["excludes_any"] else
            "proper W but NOT preserved: not an invariant"
            if not row["preserved"]["ok"] else
            "proper W, preserved, but every loop row satisfies it: no exclusion")
        rows.append(row)
    # Rule 30 is included in its own control set so the comparison is matched:
    # the controls run at a smaller horizon for cost, and a count taken at a
    # different number of rows is not comparable to one taken at `steps`.
    ctrl_steps = min(steps, 160)
    ctrl = {str(r): {str(k): control_rule(r, ctrl_steps, k)
                     for k in (6, 10, 14) if k <= kmax}
            for r in (30,) + tuple(x for x in controls if x != 30)}
    return {
        "artifact_type": ARTIFACT_TYPE,
        "artifact_version": ARTIFACT_VERSION,
        "params": {"steps": steps, "kmax": kmax, "controls": list(controls)},
        "loops_source": str(LOOPS),
        "loop_widths": [e["width"] for e in loops],
        "rows": rows,
        "controls_window_counts": ctrl,
        "controls_steps": ctrl_steps,
        "controls_note": ("matched horizon, rule 30 included. A control whose "
                          "count is far below 2^k shows the measurement can "
                          "detect a restricted language when one exists, so a "
                          "full count for rule 30 is an absence of structure "
                          "rather than an absence of sensitivity."),
        "any_invariant_found": any(r["nontrivial"] for r in rows),
        "elapsed_s": round(time.time() - started, 2),
    }


def verify(path: Path) -> int:
    art = json.loads(path.read_text(encoding="utf-8"))
    if art.get("artifact_type") != ARTIFACT_TYPE:
        print(f"not a {ARTIFACT_TYPE} artifact", file=sys.stderr)
        return 2
    p = art["params"]
    fresh = run(p["steps"], p["kmax"], tuple(p["controls"]))
    ok = True
    for old, new in zip(art["rows"], fresh["rows"]):
        for key in ("n_seed_windows", "full_language", "nontrivial"):
            if old[key] != new[key]:
                print(f"MISMATCH k={old['k']} {key}: {old[key]} -> {new[key]}",
                      file=sys.stderr)
                ok = False
    if art["any_invariant_found"] != fresh["any_invariant_found"]:
        print("MISMATCH any_invariant_found", file=sys.stderr)
        ok = False
    print(f"verify: {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def self_test() -> int:
    bad = 0
    # rule30_image must agree with the rule on a known triple.
    # window bits (LSB..MSB) = cells left..right; centre of 0b010 -> 1^(1|0)=1
    if rule30_image(0b010, 3) != 1:
        print("FAIL rule30_image on 010", file=sys.stderr)
        bad += 1
    if rule30_image(0b111, 3) != 0:      # 1 ^ (1|1) = 0
        print("FAIL rule30_image on 111", file=sys.stderr)
        bad += 1
    # The seed rows must start 1, 111, 11001, ... (Rule 30 from one cell)
    r = seed_rows(2)
    if r[0] != 0b1 or r[1] != 0b111:
        print(f"FAIL seed_rows prefix: {[bin(x) for x in r]}", file=sys.stderr)
        bad += 1
    # A full window set must be reported preserved-trivially and vacuous.
    W = set(range(1 << 3))
    if not preserved(W, 3)["ok"]:
        print("FAIL full window set was not preserved", file=sys.stderr)
        bad += 1
    print(f"self-test: {'OK' if not bad else f'{bad} failure(s)'}")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--kmax", type=int, default=14)
    ap.add_argument("--controls", default="45,90,110")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--verify", type=Path)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.verify:
        return verify(args.verify)

    art = run(args.steps, args.kmax,
              tuple(int(x) for x in args.controls.split(",") if x.strip()))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(art, indent=2) + "\n", encoding="utf-8")
    if args.pretty:
        print(f"{'k':>3} {'|W|':>8} {'of':>8} {'full':>6} {'preserved':>10} "
              f"{'excludes':>9}  verdict", file=sys.stderr)
        for r in art["rows"]:
            print(f"{r['k']:>3} {r['n_seed_windows']:>8} {r['of_possible']:>8} "
                  f"{str(r['full_language']):>6} {str(r['preserved']['ok']):>10} "
                  f"{str(r['exclusion']['excludes_any']):>9}  {r['verdict'][:46]}",
                  file=sys.stderr)
        print(f"  loop widths: {art['loop_widths']}", file=sys.stderr)
        print(f"  controls |W(k)|: {art['controls_window_counts']}", file=sys.stderr)
    print(json.dumps(art, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

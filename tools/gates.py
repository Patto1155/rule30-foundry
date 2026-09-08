#!/usr/bin/env python
"""Executable gates for experiments: the repo's rules as code, not prose.

Every rule in CLAUDE.md that has cost this repo months is currently a
paragraph an agent is asked to remember. That is adequate for a reviewer,
whose output is advice. It is not adequate for anything that *runs*
experiments, whose output enters the repo: a certificate was retracted in
2026-08 for a search class below the counting bound, and experiments I-L were
invalidated by a bit-order bug that no aggregate check could see. Neither
failure was a shortage of compute. Both were experiments that ran when they
should have been refused, or whose results were read when they should have
been rejected.

So this module turns each rule into a check that a runner must pass:

  preflight   run on an experiment MANIFEST, before any compute is spent.
              Refuses experiments that cannot produce information.
  postflight  run on an experiment RESULT, before it is written up.
              Rejects results that state more than they measured.

Every gate delegates to the tool that already owns the rule, rather than
reimplementing it: the counting bound is experiments/counting_bound.verdict,
the light cone is gpu/tape_geometry.check, bit order is tools/lint_bitorder,
the packing convention is gen_golden_reference --self-test. A gate that
re-derived the arithmetic could drift from the tool it claims to enforce.

Manifest (JSON). Fields marked * are required; the rest depend on `kind`.

    {
      "name":        * "pattern-map-walk-32",
      "kind":        * "search" | "measurement" | "simulation",
      "seed":        * "single-black-cell",
      "theory_gate": * "OPEN" | "ALREADY SETTLED" | "ROUTE CLOSED" | "NOT COVERED",
      "script":      * "experiments/pattern_map_walk.py",
      "argv":          ["--max-d", "12000000000"],
      "claims":        ["negative"],          -- conclusions the run may draw
      "reads_packed_bitstream": false,
      "search":        {"class": "dfao", "states": 24, "base": 2,
                        "prefix_bits": 10000}          -- kind == search
                       or {"class": "<other>", "log2_size": 300.0,
                           "prefix_bits": 10000}       -- non-DFAO classes
      "simulation":    {"cells": 92000064, "steps": 46000000}  -- kind == simulation
      "budget":        {"minutes": 30, "device": "cpu"}
    }

Result (JSON) for postflight. `manifest` is echoed; the rest is what ran.

    {
      "manifest":    {...},
      "horizon":     12000000000,             -- how far the run actually looked
      "metrics":     {"name": {"value": 0.003, "baseline": 0.0021}},
      "conclusions": ["No 32->64 doubling observed through d = 1.2e10."],
      "divergence":  [{"distance": 40, "first_divergence": 40}],
      "stream_comparison": {"fraction_differing": 0.4995}
    }

Usage:
    python tools/gates.py preflight  queue/b1-pattern-map-walk.json --pretty
    python tools/gates.py postflight data/wedge/result.json --pretty
    python tools/gates.py preflight  queue/trap-vacuous-dfao.json --expect-fail

`--expect-fail` inverts the exit code: it is how verify_all checks that the
trap manifest is still refused. A gate that stops gating is the failure mode.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

KINDS = ("search", "measurement", "simulation")
REQUIRED = ("name", "kind", "seed", "theory_gate", "script")
THE_SEED = "single-black-cell"

# A conclusion that says "never" without a horizon is the right-censoring
# error AGENTS.md names explicitly. These qualifiers make it honest.
CENSOR_QUALIFIERS = re.compile(
    r"\b(within|through|up to|below|before|for d\s*[<≤]|for n\s*[<≤]|"
    r"<=?\s*\d|≤\s*\d|first \d|the first)\b", re.I)
UNQUALIFIED_NEVER = re.compile(r"\b(never|no period|aperiodic|does not repeat)\b", re.I)

# The ~50% rule. A difference this close to a coin flip is uncorrelated
# streams -- packing or seed mismatch -- never a kernel bug, which diverges late.
FIFTY_PERCENT_BAND = (0.45, 0.55)


def _load(path: str):
    """Import a repo module by file path. experiments/ and gpu/ have no
    __init__.py, and adding one for this would be scope creep."""
    full = REPO_ROOT / path
    spec = importlib.util.spec_from_file_location(full.stem, full)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Gate:
    __slots__ = ("name", "status", "reason")

    def __init__(self, name: str, status: str, reason: str):
        self.name, self.status, self.reason = name, status, reason

    def as_dict(self) -> dict:
        return {"gate": self.name, "status": self.status, "reason": self.reason}


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------

def gate_schema(m: dict) -> Gate:
    missing = [k for k in REQUIRED if k not in m]
    if missing:
        return Gate("schema", FAIL, f"missing required field(s): {', '.join(missing)}")
    if m["kind"] not in KINDS:
        return Gate("schema", FAIL, f"kind must be one of {KINDS}, got {m['kind']!r}")
    return Gate("schema", PASS, "all required fields present")


def gate_seed(m: dict) -> Gate:
    """CLAUDE.md rule 3. All three prizes concern one deterministic initial
    condition. An ensemble or random-IC quantity is not prize progress,
    however well measured, so it is refused rather than run."""
    if m.get("seed") == THE_SEED:
        return Gate("seed", PASS, "single-black-cell")
    return Gate("seed", FAIL,
                f"seed is {m.get('seed')!r}, not {THE_SEED!r}. An ensemble or "
                "random-IC quantity is not progress on any prize problem "
                "(docs/theory/README.md §0).")


def gate_theory(m: dict) -> Gate:
    """AGENTS.md 'Before Proposing Experiments'. Declarative -- the manifest
    states the theory-gate verdict rather than this code deriving it from
    prose -- but mandatory, so the question cannot go unasked. Only OPEN may
    run; re-measuring a Theorem is never legitimate and re-walking a closed
    route is waste."""
    v = str(m.get("theory_gate", "")).strip().upper()
    if v == "OPEN":
        return Gate("theory-gate", PASS, "OPEN")
    if v in ("ALREADY SETTLED", "ROUTE CLOSED"):
        return Gate("theory-gate", FAIL,
                    f"theory gate says {v}: docs/theory/README.md already "
                    "answers this or has closed the route. Cite the section "
                    "and do not run.")
    return Gate("theory-gate", FAIL,
                f"theory gate is {v or 'unset'!r}; must be OPEN. Run the "
                "theory-gate check first and record its verdict.")


def gate_counting_bound(m: dict) -> Gate:
    """CLAUDE.md rule 1. A negative from class M over n bits is information
    only when log2|M| >= n; below that every sequence gives the same negative
    and the run measures |M|, not Rule 30. Equality is informative -- the
    tool's own threshold is `margin >= 0`, and Experiment S sits there."""
    if m.get("kind") != "search":
        return Gate("counting-bound", SKIP, "not a search")
    if "negative" not in (m.get("claims") or []):
        return Gate("counting-bound", SKIP, "search does not claim a negative")
    s = m.get("search") or {}
    n = s.get("prefix_bits")
    if not isinstance(n, int) or n < 1:
        return Gate("counting-bound", FAIL, "search.prefix_bits (n) is missing")

    if s.get("class") == "dfao":
        states, base = s.get("states"), s.get("base", 2)
        if not isinstance(states, int) or states < 1:
            return Gate("counting-bound", FAIL, "search.states is missing")
        cb = _load("experiments/counting_bound.py")
        v = cb.verdict(states, n, base)
        log2m = v["log2_behaviours_upper"]
        if v["informative"]:
            return Gate("counting-bound", PASS,
                        f"log2|M| = {log2m:.1f} >= n = {n} "
                        f"(margin {log2m - n:+.1f})")
        return Gate("counting-bound", FAIL,
                    f"VACUOUS: log2|M| = {log2m:.1f} < n = {n} for "
                    f"{states}-state base-{base} DFAO. {v['reading']}. "
                    "The negative is guaranteed by counting alone.")

    log2m = s.get("log2_size")
    if not isinstance(log2m, (int, float)):
        return Gate("counting-bound", FAIL,
                    f"class {s.get('class')!r} is not a DFAO and no "
                    "search.log2_size was supplied. A guess here defeats the "
                    "gate; give a defensible bound or do not claim a negative.")
    if log2m >= n:
        return Gate("counting-bound", PASS,
                    f"log2|M| = {log2m:.1f} >= n = {n} (declared)")
    return Gate("counting-bound", FAIL,
                f"VACUOUS: declared log2|M| = {log2m:.1f} < n = {n}")


def gate_light_cone(m: dict) -> Gate:
    """COMPUTE_PLAN.md §5 item 2. A tape too short for its step count does not
    crash, keeps a 0.5 bit mean, passes the first-20-bit check, and is wrong
    late. gpu/tape_geometry.check is the authority."""
    if m.get("kind") != "simulation":
        return Gate("light-cone", SKIP, "not a simulation")
    sim = m.get("simulation") or {}
    cells, steps = sim.get("cells"), sim.get("steps")
    if not all(isinstance(x, int) and x > 0 for x in (cells, steps)):
        return Gate("light-cone", FAIL, "simulation.cells and .steps required")
    tg = _load("gpu/tape_geometry.py")
    try:
        info = tg.check(cells, steps)
    except tg.ConeTooLarge as exc:
        return Gate("light-cone", FAIL, str(exc).splitlines()[0])
    return Gate("light-cone", PASS,
                f"{steps:,} steps on {info['rounded_cells']:,} cells; "
                f"{info['max_safe_steps']:,} exact")


def gate_script(m: dict) -> Gate:
    p = REPO_ROOT / str(m.get("script", ""))
    if not p.is_file():
        return Gate("script", FAIL, f"{m.get('script')} does not exist")
    if p.suffix != ".py":
        return Gate("script", FAIL, f"{m.get('script')} is not a Python script")
    return Gate("script", PASS, str(m["script"]))


def gate_bitorder_lint(m: dict, run_external: bool) -> Gate:
    """CLAUDE.md rule 2. A bare np.unpackbits reverses every 8-bit block."""
    if not m.get("reads_packed_bitstream"):
        return Gate("bitorder-lint", SKIP, "does not read a packed bitstream")
    if not run_external:
        return Gate("bitorder-lint", SKIP, "external checks disabled")
    r = subprocess.run([sys.executable, "tools/lint_bitorder.py", "--quiet"],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    if r.returncode == 0:
        return Gate("bitorder-lint", PASS, "no bare packbits/unpackbits")
    return Gate("bitorder-lint", FAIL, "bare np.packbits/unpackbits in repo; "
                "run tools/lint_bitorder.py")


def gate_golden(m: dict, run_external: bool) -> Gate:
    """The packing convention itself: naive == packed and the OEIS A051023
    prefix matches. If this fails, no bitstream on this machine is trusted."""
    if not m.get("reads_packed_bitstream"):
        return Gate("golden-self-test", SKIP, "does not read a packed bitstream")
    if not run_external:
        return Gate("golden-self-test", SKIP, "external checks disabled")
    r = subprocess.run([sys.executable, "tools/gen_golden_reference.py",
                        "--self-test"], cwd=REPO_ROOT,
                       capture_output=True, text=True)
    if r.returncode == 0:
        return Gate("golden-self-test", PASS, "naive == packed, OEIS prefix OK")
    return Gate("golden-self-test", FAIL,
                (r.stdout + r.stderr).strip().splitlines()[-1:][0]
                if (r.stdout + r.stderr).strip() else "self-test failed")


def gate_verify_all(run_external: bool) -> Gate:
    """COMPUTE_PLAN.md §5 item 1: verify_all before spending anything.
    Permissive mode here (SKIPs allowed) because a fresh clone legitimately
    lacks the bitstreams; a FAIL is still a FAIL."""
    if not run_external:
        return Gate("verify-all", SKIP, "external checks disabled")
    r = subprocess.run([sys.executable, "tools/verify_all.py"],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or ["(no output)"])
    summary = next((ln for ln in reversed(last) if ln.startswith("verify_all:")),
                   last[-1])
    return Gate("verify-all", PASS if r.returncode == 0 else FAIL, summary)


def preflight(m: dict, run_external: bool = True) -> dict:
    """All preflight gates. `run_external=False` skips the subprocess gates so
    the pure logic can be unit-tested in milliseconds; the CLI always runs
    them."""
    gates = [gate_schema(m)]
    if gates[0].status == FAIL:
        return _report("preflight", m.get("name"), gates)
    gates += [
        gate_seed(m),
        gate_theory(m),
        gate_counting_bound(m),
        gate_light_cone(m),
        gate_script(m),
        gate_bitorder_lint(m, run_external),
        gate_golden(m, run_external),
        gate_verify_all(run_external),
    ]
    return _report("preflight", m.get("name"), gates)


# --------------------------------------------------------------------------
# postflight
# --------------------------------------------------------------------------

def unqualified_clauses(text: str) -> list[str]:
    """Clauses that assert 'never' without a horizon, judged per clause.

    Matching over a whole conclusion lets an unrelated bounded phrase suppress
    an absolute claim later in the same string: "Within 10 steps the seed
    settled; the sequence never repeats." would pass, which is exactly the
    unqualified right-censored claim this gate exists to reject. Compound
    prose of that shape is natural for a result writer, so the qualifier has
    to sit in the same clause as the claim it qualifies.
    """
    bad = []
    for clause in re.split(r"[.;]+\s*", text):
        if UNQUALIFIED_NEVER.search(clause) and not CENSOR_QUALIFIERS.search(clause):
            bad.append(clause.strip())
    return bad


def gate_censoring(r: dict) -> Gate:
    """AGENTS.md: 'Never reached within N steps' is right-censored. Do not
    report it as 'never' without qualification."""
    bad = []
    for c in r.get("conclusions") or []:
        bad.extend(unqualified_clauses(c))
    if bad:
        return Gate("censoring", FAIL,
                    "unqualified 'never'/'no period' in: "
                    + " | ".join(f"{b[:70]}…" if len(b) > 70 else b for b in bad)
                    + ". State the horizon: 'not observed through N'.")
    kind = (r.get("manifest") or {}).get("kind")
    if kind in ("search", "simulation") and "horizon" not in r:
        return Gate("censoring", FAIL,
                    f"a {kind} result must state `horizon` (how far it looked)")
    return Gate("censoring", PASS, "every conclusion is horizon-qualified")


def gate_noise_floor(r: dict) -> Gate:
    """AGENTS.md: if a metric is near zero, define a noise floor or baseline
    before calling it asymmetric or structured. Mechanical form: every
    reported metric carries a measured baseline."""
    metrics = r.get("metrics") or {}
    missing = [k for k, v in metrics.items()
               if not isinstance(v, dict) or "baseline" not in v]
    if missing:
        return Gate("noise-floor", FAIL,
                    f"metric(s) without a measured baseline: {', '.join(missing)}")
    return Gate("noise-floor", PASS,
                f"{len(metrics)} metric(s), each with a baseline")


def gate_divergence(r: dict) -> Gate:
    """AGENTS.md: in a radius-1 CA, first_divergence < distance is impossible.
    Treat it as a hard failure, not a curiosity."""
    for d in r.get("divergence") or []:
        try:
            if d["first_divergence"] < d["distance"]:
                return Gate("divergence-invariant", FAIL,
                            f"first_divergence {d['first_divergence']} < "
                            f"distance {d['distance']}: impossible for a "
                            "radius-1 CA. This is a bug, not a result.")
        except (KeyError, TypeError):
            return Gate("divergence-invariant", FAIL,
                        "divergence entries need distance and first_divergence")
    return Gate("divergence-invariant", PASS, "no impossible divergence")


def gate_fifty_percent(r: dict) -> Gate:
    """AGENTS.md: a ~50% bit difference between two streams is never a kernel
    bug -- it means they are uncorrelated, i.e. packing or seed mismatch."""
    sc = r.get("stream_comparison")
    if not sc:
        return Gate("fifty-percent", SKIP, "no stream comparison reported")
    f = sc.get("fraction_differing")
    if not isinstance(f, (int, float)):
        return Gate("fifty-percent", FAIL, "fraction_differing missing")
    lo, hi = FIFTY_PERCENT_BAND
    if lo <= f <= hi:
        return Gate("fifty-percent", FAIL,
                    f"{f:.4f} of positions differ: the streams are "
                    "uncorrelated. Packing or seed mismatch -- not a kernel "
                    "bug, which diverges late. Do not investigate the kernel.")
    return Gate("fifty-percent", PASS, f"{f:.4f} differing, outside the "
                "uncorrelated band")


def gate_seed_echo(r: dict) -> Gate:
    m = r.get("manifest") or {}
    if m.get("seed") == THE_SEED:
        return Gate("seed-echo", PASS, THE_SEED)
    return Gate("seed-echo", FAIL, "result does not echo the single-black-cell "
                "seed in its manifest")


def postflight(r: dict) -> dict:
    gates = [
        gate_seed_echo(r),
        gate_censoring(r),
        gate_noise_floor(r),
        gate_divergence(r),
        gate_fifty_percent(r),
    ]
    return _report("postflight", (r.get("manifest") or {}).get("name"), gates)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _report(stage: str, name, gates: list[Gate]) -> dict:
    verdict = FAIL if any(g.status == FAIL for g in gates) else PASS
    return {
        "artifact_type": f"rule30.gates.{stage}",
        "name": name,
        "verdict": verdict,
        "gates": [g.as_dict() for g in gates],
    }


def pretty(report: dict) -> str:
    width = max(len(g["gate"]) for g in report["gates"])
    lines = [f"{report['artifact_type']}  {report.get('name') or ''}"]
    for g in report["gates"]:
        lines.append(f"  {g['status']:<4}  {g['gate']:<{width}}  {g['reason']}")
    lines.append(f"  --> {report['verdict']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("preflight", "postflight"))
    ap.add_argument("path", type=Path, help="manifest (preflight) or result (postflight) JSON")
    ap.add_argument("--pretty", action="store_true", help="human table on stderr")
    ap.add_argument("--no-external", action="store_true",
                    help="skip subprocess gates (lint, golden, verify_all)")
    ap.add_argument("--expect-fail", action="store_true",
                    help="exit 0 iff the verdict is FAIL; for verify_all's trap check")
    args = ap.parse_args(argv)

    try:
        doc = json.loads(args.path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"gates: cannot read {args.path}: {exc}", file=sys.stderr)
        return 2

    report = (preflight(doc, run_external=not args.no_external)
              if args.stage == "preflight" else postflight(doc))
    print(json.dumps(report, indent=2))
    if args.pretty:
        print(pretty(report), file=sys.stderr)

    failed = report["verdict"] == FAIL
    if args.expect_fail:
        if not failed:
            print(f"gates: expected {args.path} to be refused, but it PASSED. "
                  "A gate has stopped gating.", file=sys.stderr)
        return 0 if failed else 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

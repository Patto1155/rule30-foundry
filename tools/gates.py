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
      "purpose":     * "prize-claim"       -- a statement about the column
                     | "exact-exclusion"   -- a finite class excluded outright
                     | "exploratory"       -- a pilot aimed at the column
                     | "correctness-check" -- a statement about the instrument
                     | "replication",      -- reproduces a prior run
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
                       or {"class": "annihilator", "window_bits": 8,
                           "degree": 2, "n_distinct_windows": 101,
                           "margin_bits": 64}          -- margin defaults to 64
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

`purpose` scopes the gates. `seed` refuses a non-single-seed run for the
three purposes that speak about the column, requires a `replication` to
preserve the seed of the run it names, and does not apply to an instrument
check. `theory-gate` and `counting-bound` do not apply to an instrument check
either: a positive control must be free to target settled ground, and a
detection-power control must be free to run a class too small to fit.
`counting-bound` also admits a declared-exhaustive `exact-exclusion` as a
bounded finding -- whether a negative DISCRIMINATES the sequence and whether
it is TRUE are separate questions, and only the first is a counting
question.
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

# What the run's output will be read AS. `kind` says how the run is shaped;
# `purpose` says what a reader is entitled to conclude from it, and the gates
# below are scoped by it. Without this axis a rule that exists to protect
# prize claims also refuses the instrument checks the workflow mandates --
# which is exactly what the blanket seed gate did to docs/WORKFLOW.md's
# "always test a random IC".
PURPOSES = ("prize-claim", "exact-exclusion", "exploratory",
            "correctness-check", "replication")

REQUIRED = ("name", "kind", "purpose", "seed", "theory_gate", "script")
THE_SEED = "single-black-cell"

# Purposes whose output is a statement about the single-seed center column,
# and which must therefore use that seed (CLAUDE.md rule 3).
SEED_SCOPED_PURPOSES = ("prize-claim", "exact-exclusion", "exploratory")

# Purposes that are statements about the instrument rather than about Rule 30.
# The initial condition is a free parameter for these, and a random one is
# positively required to expose open-boundary padding bugs. `replication` is
# deliberately NOT here: it is free of the single-seed rule but not free of
# the seed it claims to reproduce -- see gate_seed.
INSTRUMENT_PURPOSES = ("correctness-check",)

# A conclusion that says "never" without a horizon is the right-censoring
# error AGENTS.md names explicitly. These qualifiers make it honest.
CENSOR_QUALIFIERS = re.compile(
    r"\b(within|through|up to|below|before|for d\s*[<≤]|for n\s*[<≤]|"
    r"<=?\s*\d|≤\s*\d|first \d|the first)\b", re.I)
UNQUALIFIED_NEVER = re.compile(r"\b(never|no period|aperiodic|does not repeat)\b", re.I)

# The ~50% rule. A difference this close to a coin flip means the two streams
# are uncorrelated. That is all it means: the rate identifies no cause, and
# ranking causes by it is a guess dressed as a diagnosis. Check packing and
# seed conventions, then localise the first divergence before assigning a
# cause -- an early-step kernel bug decorrelates everything after it and
# lands in this band exactly as a packing mismatch does.
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
    if m["purpose"] not in PURPOSES:
        return Gate("schema", FAIL,
                    f"purpose must be one of {PURPOSES}, got {m['purpose']!r}. "
                    "It decides which gates apply, so it cannot be guessed.")
    return Gate("schema", PASS,
                f"all required fields present; purpose {m['purpose']!r}")


def gate_seed(m: dict) -> Gate:
    """CLAUDE.md rule 3, scoped by `purpose`. All three prizes concern one
    deterministic initial condition, so an ensemble or random-IC quantity is
    not prize progress however well measured, and is refused rather than run.

    It is refused only for the purposes that make a statement about the
    center column. docs/WORKFLOW.md requires a random IC to catch packed
    open-boundary padding bugs -- they leave the edges at 0, so the single
    seed cannot see them -- and an unscoped seed gate refuses that check,
    which is the conflict `purpose` exists to resolve. A correctness check is
    not weaker evidence about the prize; it is evidence about the instrument,
    and the ledger never reads it as anything else.
    """
    purpose = m.get("purpose")

    if purpose == "replication":
        # Exempting a replication from the seed rule outright would let a
        # random-IC run claim to reproduce a single-seed result: the strongest
        # possible false positive, since a replication's whole value is that
        # it reproduces a specific prior run. So the rule is not waived, it is
        # redirected -- the seed must match the source's, and the source's
        # seed must be stated where a reviewer can check it against the log.
        rep = m.get("replicates")
        if not isinstance(rep, dict):
            return Gate("seed", FAIL,
                        "purpose 'replication' requires a `replicates` object "
                        "naming what is being reproduced: "
                        '{"source": "<log or artifact path>", "seed": "<the '
                        'seed that run used>"}.')
        source, src_seed = rep.get("source"), rep.get("seed")
        if not isinstance(source, str) or not source.strip():
            return Gate("seed", FAIL, "replicates.source is missing")
        if not isinstance(src_seed, str) or not src_seed.strip():
            return Gate("seed", FAIL,
                        f"replicates.seed is missing for source {source!r}. "
                        "State the seed the original run used; an unstated "
                        "one cannot be checked against this manifest's.")
        if m.get("seed") != src_seed:
            return Gate("seed", FAIL,
                        f"replication seed {m.get('seed')!r} does not match "
                        f"the seed {src_seed!r} of the run it claims to "
                        f"reproduce ({source}). A different initial condition "
                        "is a different experiment, not a replication.")
        return Gate("seed", PASS,
                    f"replication of {source} preserves its seed {src_seed!r}")

    if purpose in INSTRUMENT_PURPOSES:
        return Gate("seed", SKIP,
                    f"purpose is {purpose!r}: the result is a statement about "
                    "the instrument, not about the single-seed center column. "
                    "Any initial condition is admissible, and a random one is "
                    "required for open-boundary checks (docs/WORKFLOW.md).")
    if m.get("seed") == THE_SEED:
        return Gate("seed", PASS, f"single-black-cell (purpose {purpose!r})")
    return Gate("seed", FAIL,
                f"seed is {m.get('seed')!r}, not {THE_SEED!r}, under purpose "
                f"{purpose!r}. An ensemble or random-IC quantity is not "
                "progress on any prize problem (docs/theory/README.md §0). If "
                "this run checks the kernel rather than the column, declare "
                "purpose 'correctness-check' instead of changing the seed.")


def gate_theory(m: dict) -> Gate:
    """AGENTS.md 'Before Proposing Experiments'. Declarative -- the manifest
    states the theory-gate verdict rather than this code deriving it from
    prose -- but mandatory, so the question cannot go unasked. Only OPEN may
    run; re-measuring a Theorem is never legitimate and re-walking a closed
    route is waste.

    That reasoning is about learning something new concerning Rule 30, so it
    does not apply to a check on the instrument. A positive control is
    *required* to target settled ground -- Thue-Morse returning s*=2 is
    valuable precisely because the answer is known in advance, and a kernel
    check reproducing a hash-anchored stream is checking the tool against an
    answer theory already fixed. Refusing those as "ALREADY SETTLED" would
    refuse every control the repo has, so this gate does not run for them."""
    purpose = m.get("purpose")
    if purpose in INSTRUMENT_PURPOSES:
        return Gate("theory-gate", SKIP,
                    f"purpose is {purpose!r}: a control checks the instrument "
                    "against a known answer, so settled ground is the point "
                    "rather than a reason to refuse it.")
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


def _gate_annihilator(s: dict) -> Gate:
    """Apply both annihilator bounds, including for positive-only searches."""
    values = {}
    for key in ("window_bits", "degree", "n_distinct_windows", "margin_bits"):
        value = s.get(key, 64 if key == "margin_bits" else None)
        if type(value) is not int:
            return Gate("counting-bound", FAIL,
                        f"search.{key} must be an integer (not bool)")
        values[key] = value
    w, d = values["window_bits"], values["degree"]
    n, margin = values["n_distinct_windows"], values["margin_bits"]
    if w < 1:
        return Gate("counting-bound", FAIL,
                    "search.window_bits must be >= 1")
    if not 0 <= d <= w:
        return Gate("counting-bound", FAIL, "search.degree must be in 0..window_bits")
    if not 1 <= n <= (1 << w):
        return Gate("counting-bound", FAIL,
                    "search.n_distinct_windows must be in 1..2^window_bits")
    if margin < 0:
        return Gate("counting-bound", FAIL, "search.margin_bits must be >= 0")
    try:
        v = _load("experiments/counting_bound.py").annihilator_verdict(w, d, n, margin)
    except (OverflowError, ValueError) as exc:
        return Gate("counting-bound", FAIL,
                    f"Unsupported annihilator parameters in counting_bound helper: {exc}")
    dim, ceiling = v["monomial_dimension"], v["max_zeros_reed_muller"]
    if n > ceiling:
        return Gate("counting-bound", FAIL, v["reading"])
    if n < dim:
        return Gate("counting-bound", FAIL,
                    f"FORCED POSITIVE: {n} distinct windows < D={dim} monomials; "
                    "a nonzero kernel exists by dimension alone.")
    if not v["informative"]:
        return Gate("counting-bound", FAIL,
                    f"SAFETY MARGIN SHORTFALL: {n} distinct windows >= D={dim}, "
                    f"but below D+margin_bits={dim + margin}. This conservative "
                    "policy refusal does not imply a forced kernel or a probability bound.")
    return Gate("counting-bound", PASS,
                f"Annihilator preflight: D={dim} <= {n} distinct windows; "
                f"D+margin_bits={dim + margin} <= {n} <= Reed-Muller ceiling {ceiling}. "
                "These necessary bounds do not establish a relation or its predictive value.")


def _non_discriminating(m: dict, s: dict, detail: str) -> Gate:
    """Verdict for a negative that a coin would also have produced.

    Two independent questions, and the old gate answered only one while
    claiming both. Whether the negative DISCRIMINATES this sequence from a
    random one is the counting bound: below the threshold, P(a uniform random
    string fits) <= 2^(log2|M| - n), so the negative carries almost no
    information about Rule 30. Whether the negative is TRUE is exhaustiveness:
    a complete search of M that finds no fit has proved no member of M
    generates the prefix, and that proof does not weaken as M shrinks.

    So a declared-exhaustive `exact-exclusion` is admitted as a bounded
    finding, with the limit stated in the verdict rather than left for the
    write-up to remember. Everything that makes a claim about the column is
    still refused: the finding is real and it is not evidence of complexity,
    and those are not the same permission.
    """
    exhaustive = s.get("exhaustive") is True
    if m.get("purpose") == "exact-exclusion":
        if not exhaustive:
            return Gate("counting-bound", FAIL,
                        f"{detail} purpose 'exact-exclusion' admits a bounded "
                        "finding only when the search is complete: declare "
                        '`"exhaustive": true` in `search`. A search that was '
                        "not exhaustive has not excluded anything -- it failed "
                        "to find something, which is the weaker claim the "
                        "counting bound refuses.")
        return Gate("counting-bound", PASS,
                    f"{detail} admitted as a BOUNDED EXCLUSION: an exhaustive "
                    "search proves no member of this class generates the "
                    "prefix, and that is true however small the class. It is "
                    "not evidence that the sequence is complex -- a coin gives "
                    "the same negative -- so state the class and do not "
                    "generalise beyond it.")
    return Gate("counting-bound", FAIL,
                f"{detail} A uniform random string would almost certainly "
                "give the same negative, so this measures |M|, not Rule 30. "
                "If the search is exhaustive, the exclusion is still true: "
                "declare purpose 'exact-exclusion' with `\"exhaustive\": true` "
                "and record it as a bounded finding rather than as evidence.")


def gate_counting_bound(m: dict) -> Gate:
    """CLAUDE.md rule 1. A negative from class M over n bits is information
    only when log2|M| >= n; below that every sequence gives the same negative
    and the run measures |M|, not Rule 30. Equality is informative -- the
    tool's own threshold is `margin >= 0`, and Experiment S sits there.

    Scoped away from instrument checks for the same reason as the theory
    gate: a detection-power control deliberately runs a class too small to
    fit, to confirm the search reports a negative when it should. Its
    negative is a measurement of the detector, not a claim about Rule 30, and
    the counting bound has nothing to say about it. This does NOT relax the
    rule for `exact-exclusion`, which still fails here -- recording a bounded
    exclusion as a finding needs the vacuity verdict split from the
    truth of the exclusion, which has not been done yet."""
    purpose = m.get("purpose")
    if purpose in INSTRUMENT_PURPOSES:
        return Gate("counting-bound", SKIP,
                    f"purpose is {purpose!r}: the negative measures the "
                    "detector, not the sequence.")
    if m.get("kind") != "search":
        return Gate("counting-bound", SKIP, "not a search")
    s = m.get("search")
    if not isinstance(s, dict):
        return Gate("counting-bound", FAIL, "search must be an object")
    if s.get("class") == "annihilator":
        return _gate_annihilator(s)
    if "negative" not in (m.get("claims") or []):
        return Gate("counting-bound", SKIP, "search does not claim a negative")
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
        return _non_discriminating(
            m, s,
            f"NOT DISCRIMINATING: log2|M| = {log2m:.1f} < n = {n} for "
            f"{states}-state base-{base} DFAO (expected fits "
            f"2^{v['log2_expected_fits']:.1f}).")

    log2m = s.get("log2_size")
    if not isinstance(log2m, (int, float)):
        return Gate("counting-bound", FAIL,
                    f"class {s.get('class')!r} is not a DFAO and no "
                    "search.log2_size was supplied. A guess here defeats the "
                    "gate; give a defensible bound or do not claim a negative.")
    if log2m >= n:
        return Gate("counting-bound", PASS,
                    f"log2|M| = {log2m:.1f} >= n = {n} (declared)")
    return _non_discriminating(
        m, s,
        f"NOT DISCRIMINATING: declared log2|M| = {log2m:.1f} < n = {n} "
        f"(expected fits 2^{log2m - n:+.1f}).")


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
    """AGENTS.md: a ~50% bit difference means the two streams are
    uncorrelated, and nothing more. Check packing and seed conventions, then
    localise the first divergence before assigning a cause --
    `divergence[].first_divergence` is the evidence that distinguishes a
    convention mismatch from an early-step kernel bug. The rate does not."""
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
                    "uncorrelated. Check packing and seed conventions, then "
                    "localise the first divergence before assigning a cause. "
                    "This rate is consistent with a convention mismatch and "
                    "with an early-step kernel bug alike, so it names neither.")
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

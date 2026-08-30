#!/usr/bin/env python
"""Minimal-DFAO-size curve s*(n) for the Rule 30 center column.

Motivation
----------
The repo previously recorded a "no 1-5 state DFAO fits the first 128 center bits"
result as a Certificate.  That statement is **vacuous**: it is implied by counting
alone for *any* binary sequence.

The bound lives in ``experiments/counting_bound.py`` (see
``docs/theory/finite-prefix-counting-bound.md``).  Briefly: a binary-output DFAO
with ``s`` states over base-``b`` digits and a fixed initial state has at most
``s**(s*b) * 2**s`` distinct behaviours, and a fixed DFAO matches a uniformly
random length-``n`` binary string with probability ``2**-n``, so by the union
bound

    P[some s-state DFAO fits n random bits] <= 2 ** (log2|M(s,b)| - n) .

A negative ("no s-state DFAO fits") therefore carries information only once
``log2|M(s,b)| >= n``.  At ``n = 128``, base 2, that needs ``s >= 15`` even on the
most optimistic accounting -- an exhaustive space of ``2**132`` -- so the 1-5
state result was never a test of Rule 30 (expected fits ``2**-99.8``).

Vacuity verdicts use the UPPER bound on ``|M|`` (overstating the class makes the
vacuity claim conservative); fair-threshold questions use the LOWER bound
obtained by quotienting the ``(s-1)!`` relabelings of the non-initial states.
The two differ, so the counting null for ``s*(n)`` is reported as a **band**.

The non-vacuous replacement measured here is the **minimal-DFAO-size curve**

    s*(n) = min { s : some s-state binary-output DFAO reproduces bits 0..n-1 }

for the Rule 30 center column, compared against a random control (the counting
null made empirical) and Thue-Morse (a genuinely automatic positive control).
If the center column has any automatic structure at all, s*(n) should sit
*below* the random/counting curve.

Implementation notes
--------------------
All sequence generation and the CNF encoding are reused unchanged from
``prize_lab.py``.  This module adds only (a) an in-process pysat solving path,
(b) a decoder from a SAT model back to a DFAO candidate, and (c) the sweep.

JSON goes to stdout, progress to stderr, matching the rest of the repo.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

from counting_bound import (  # noqa: E402
    dfao_table,
    fair_thresholds,
    log2_dfao_lower,
    log2_dfao_upper,
    verdict as counting_verdict,
)

from prize_lab import (  # noqa: E402
    ARTIFACT_VERSION,
    bits_ascii,
    dfao_sat_cnf,
    exhaustive_dfao_search,
    git_context,
    relative_to_root,
    run_dfao,
    sequence_bits,
    sha256_ascii,
    utc_now_iso,
    verify_candidate_on_bits,
    write_json,
)

PYSAT_HINT = "python-sat is required: pip install python-sat  (import name 'pysat')"

# Do NOT raise at import time. This is an availability probe only -- the solver
# is imported locally in the worker (see solve_cnf_worker). Raising SystemExit
# here was a live defect: SystemExit does not inherit from Exception, so an
# `except ImportError`/`except Exception` guard in an importing module cannot
# catch it, and a SystemExit raised during `unittest discover` tears down the
# whole test process, so every module later in discovery order is never
# collected. main() turns the missing dependency into a clean CLI error.
try:
    from pysat.solvers import Cadical153  # noqa: F401  (availability check only)
    PYSAT_AVAILABLE = True
except ImportError:  # pragma: no cover - environment guard
    PYSAT_AVAILABLE = False


SOLVER_NAME = "Cadical153"

# prize_lab.CnfBuilder variable names
TRANS_RE = re.compile(r"t_state(\d+)_digit(\d+)_state(\d+)")
OUT_RE = re.compile(r"out_state(\d+)")


def log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# counting null (delegates to experiments/counting_bound.py)
# ---------------------------------------------------------------------------


def counting_null_band(n_bits: int, base: int, *, search_cap: int = 512) -> dict:
    """Smallest s whose DFAO class could plausibly fit n bits, as a band.

    ``optimistic`` uses the upper bound on |M| (smallest s that could suffice);
    ``conservative`` uses the lower bound. The true counting threshold lies
    between them, so the null is a band, not a line.
    """
    s_opt = next((s for s in range(1, search_cap) if log2_dfao_upper(s, base) >= n_bits), None)
    s_cons = next((s for s in range(1, search_cap) if log2_dfao_lower(s, base) >= n_bits), None)
    return {"optimistic": s_opt, "conservative": s_cons}


def expected_random_fits_log2(states: int, base: int, n_bits: int) -> float:
    """log2 of the union-bound expected number of s-state DFAOs fitting n random bits."""
    return log2_dfao_upper(states, base) - n_bits


# ---------------------------------------------------------------------------
# DIMACS -> clause list + variable map
# ---------------------------------------------------------------------------


def parse_dimacs(text: str) -> tuple[list[list[int]], dict[int, str], int, int]:
    """Parse the DIMACS emitted by prize_lab.dfao_sat_cnf.

    prize_lab writes a ``c var <id> <name>`` comment for every variable, so the
    text carries its own symbol table -- no re-derivation of the encoding needed.
    """
    clauses: list[list[int]] = []
    names: dict[int, str] = {}
    n_vars = 0
    declared_clauses = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("c"):
            parts = stripped.split()
            if len(parts) == 4 and parts[1] == "var":
                names[int(parts[2])] = parts[3]
            continue
        if stripped.startswith("p "):
            parts = stripped.split()
            if len(parts) != 4 or parts[1] != "cnf":
                raise ValueError(f"unsupported DIMACS header: {stripped}")
            n_vars = int(parts[2])
            declared_clauses = int(parts[3])
            continue
        lits = [int(tok) for tok in stripped.split()]
        if lits and lits[-1] == 0:
            lits.pop()
        clauses.append(lits)
    if n_vars == 0:
        raise ValueError("missing DIMACS header")
    if declared_clauses != len(clauses):
        raise ValueError(
            f"DIMACS header claims {declared_clauses} clauses, parsed {len(clauses)}"
        )
    if len(names) != n_vars:
        raise ValueError(f"symbol table has {len(names)} entries for {n_vars} variables")
    return clauses, names, n_vars, declared_clauses


def decode_model(
    model: list[int],
    names: dict[int, str],
    *,
    states: int,
    base: int,
    direction: str,
) -> dict:
    """Turn a satisfying assignment back into a DFAO candidate artifact.

    Variable names come from prize_lab's CnfBuilder:
      x_node{node}_state{state}          node of the digit trie sits in state
      t_state{a}_digit{d}_state{dst}     transition a --d--> dst
      out_state{s}                       output label of state s
    """
    true_vars = {abs(lit) for lit in model if lit > 0}
    transitions: list[list[int | None]] = [[None] * base for _ in range(states)]
    outputs = [0] * states
    for var, name in names.items():
        if var not in true_vars:
            continue
        m = TRANS_RE.fullmatch(name)
        if m:
            src, digit, dst = (int(g) for g in m.groups())
            if transitions[src][digit] is not None:
                raise ValueError(f"model sets two transitions for state {src} digit {digit}")
            transitions[src][digit] = dst
            continue
        m = OUT_RE.fullmatch(name)
        if m:
            outputs[int(m.group(1))] = 1
    for src in range(states):
        for digit in range(base):
            if transitions[src][digit] is None:
                raise ValueError(f"model left transition ({src},{digit}) unset")
    return {
        "artifact_type": "rule30.dfao_candidate",
        "artifact_version": ARTIFACT_VERSION,
        "states": states,
        "base": base,
        "direction": direction,
        "initial_state": 0,
        "transitions": transitions,
        "outputs": outputs,
    }


# ---------------------------------------------------------------------------
# in-process solving
# ---------------------------------------------------------------------------


def solve_cnf_worker(cnf_path: str, result_path: str) -> int:
    """Child-process entry point: solve a DIMACS file, write the verdict as JSON.

    Solving happens in a child process so that the wall-clock timeout can be a
    hard kill.  pysat's ``Cadical153.interrupt()`` raises ``NotImplementedError``,
    so an in-process timer cannot actually stop a running solve -- an earlier
    version of this script silently ran past its stated timeout because of that.
    A killed child is unambiguously UNKNOWN: no result file is ever written.
    """
    from pysat.solvers import Cadical153 as _Cadical  # local import for the child

    clauses, _names, _n_vars, _ = parse_dimacs(Path(cnf_path).read_text(encoding="ascii"))
    solver = _Cadical(bootstrap_with=clauses)
    result = solver.solve()
    payload = {
        "status": "SAT" if result else "UNSAT",
        "model": solver.get_model() if result else None,
    }
    solver.delete()
    Path(result_path).write_text(json.dumps(payload), encoding="ascii")
    return 0


def solve_dfao_sat(
    bits: list[int],
    *,
    states: int,
    base: int,
    direction: str,
    timeout_s: float,
) -> dict:
    """Does an s-state DFAO reproduce ``bits``?  SAT / UNSAT / UNKNOWN (timeout).

    UNKNOWN means the solver was killed at the wall-clock limit. It is never
    reported as UNSAT.
    """
    t_build = time.perf_counter()
    dimacs, meta = dfao_sat_cnf(bits, states=states, base=base, direction=direction)
    clauses, names, n_vars, _ = parse_dimacs(dimacs)
    build_s = time.perf_counter() - t_build

    tmpdir = tempfile.mkdtemp(prefix="dfao_sat_")
    cnf_path = Path(tmpdir) / "instance.cnf"
    result_path = Path(tmpdir) / "result.json"
    cnf_path.write_text(dimacs, encoding="ascii")

    t0 = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--solve-cnf", str(cnf_path), "--solve-result", str(result_path)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if proc.returncode != 0 and not result_path.exists():
            raise RuntimeError(
                f"solver child failed (rc={proc.returncode}): {proc.stderr[-2000:]}"
            )
    except subprocess.TimeoutExpired:
        timed_out = True
    solve_s = time.perf_counter() - t0

    if timed_out or not result_path.exists():
        status = "UNKNOWN"
        model = None
    else:
        payload = json.loads(result_path.read_text(encoding="ascii"))
        status = payload["status"]
        model = payload["model"]

    for p in (cnf_path, result_path):
        try:
            p.unlink()
        except OSError:
            pass
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass

    out = {
        "status": status,
        "timed_out": bool(timed_out),
        "solve_s": round(solve_s, 6),
        "build_s": round(build_s, 6),
        "variables": n_vars,
        "clauses": len(clauses),
        "trie_nodes": meta["trie_nodes"],
        "candidate": None,
        "candidate_verified": None,
    }
    if status == "SAT":
        candidate = decode_model(model, names, states=states, base=base, direction=direction)
        check = verify_candidate_on_bits(candidate, bits)
        out["candidate"] = candidate
        out["candidate_verified"] = check
        if not check["ok"]:
            raise RuntimeError(
                f"decoded DFAO does not reproduce the bits it was solved for: {check}"
            )
    return out


# ---------------------------------------------------------------------------
# correctness gate: SAT path vs exhaustive path
# ---------------------------------------------------------------------------


def run_correctness_gate(*, timeout_s: float, verbose: bool = True) -> dict:
    """Mandatory pre-science gate. Any disagreement here is worth more than the sweep."""
    checks: list[dict] = []

    def record(name: str, ok: bool, detail: dict) -> None:
        checks.append({"check": name, "ok": bool(ok), **detail})
        if verbose:
            log(f"  gate {'PASS' if ok else 'FAIL'}  {name}")

    # 1. Thue-Morse positive control: SAT at s=2, decoded automaton must run correctly.
    tm = sequence_bits("thue-morse", 64)
    res = solve_dfao_sat(tm, states=2, base=2, direction="msd", timeout_s=timeout_s)
    ok = res["status"] == "SAT" and res["candidate_verified"]["ok"]
    ok = ok and all(run_dfao(res["candidate"], n) == b for n, b in enumerate(tm))
    record(
        "thue-morse n=64 msd s=2 is SAT and decoded DFAO reproduces bits",
        ok,
        {"status": res["status"], "candidate": res["candidate"]},
    )

    # 1b. same, LSD.
    res_lsd = solve_dfao_sat(tm, states=2, base=2, direction="lsd", timeout_s=timeout_s)
    ok = res_lsd["status"] == "SAT" and res_lsd["candidate_verified"]["ok"]
    record("thue-morse n=64 lsd s=2 is SAT", ok, {"status": res_lsd["status"]})

    # 2. Thue-Morse at s=1 must be UNSAT (the sequence is not constant).
    res1 = solve_dfao_sat(tm, states=1, base=2, direction="msd", timeout_s=timeout_s)
    record("thue-morse n=64 msd s=1 is UNSAT", res1["status"] == "UNSAT", {"status": res1["status"]})

    # 3. SAT vs exhaustive agreement wherever exhaustive is feasible (s <= 4, base 2).
    agreements: list[dict] = []
    disagreements: list[dict] = []
    for sequence, seed in (("center", 0), ("thue-morse", 0), ("random", 30)):
        for n in (8, 12, 16, 20, 24):
            for direction in ("msd", "lsd"):
                bits = sequence_bits(sequence, n, seed=seed)
                # exhaustive: smallest s <= 4 that fits, or None
                ex = exhaustive_dfao_search(
                    bits,
                    max_states=4,
                    base=2,
                    direction=direction,
                    max_transitions=100_000_000,
                    progress=False,
                )
                ex_s = ex["candidate"]["states"] if ex["found"] else None
                for s in range(1, 5):
                    sat = solve_dfao_sat(bits, states=s, base=2, direction=direction, timeout_s=timeout_s)
                    sat_fits = sat["status"] == "SAT"
                    # exhaustive says "fits with <= s states" iff ex_s is not None and ex_s <= s
                    ex_fits = ex_s is not None and ex_s <= s
                    row = {
                        "sequence": sequence,
                        "n": n,
                        "direction": direction,
                        "states": s,
                        "sat_status": sat["status"],
                        "exhaustive_fits": ex_fits,
                    }
                    if sat["status"] == "UNKNOWN":
                        disagreements.append({**row, "reason": "sat path timed out"})
                    elif sat_fits != ex_fits:
                        disagreements.append({**row, "reason": "verdict mismatch"})
                    else:
                        agreements.append(row)
    record(
        f"SAT vs exhaustive agree on {len(agreements)} (sequence,n,direction,s) cells, s<=4",
        not disagreements,
        {"agreements": len(agreements), "disagreements": disagreements},
    )

    passed = all(c["ok"] for c in checks)
    if verbose:
        log(f"  gate overall: {'PASS' if passed else 'FAIL'}")
    return {"passed": passed, "checks": checks}


# ---------------------------------------------------------------------------
# the curve
# ---------------------------------------------------------------------------


def min_states_for_prefix(
    bits: list[int],
    *,
    base: int,
    direction: str,
    max_states: int,
    timeout_s: float,
    label: str = "",
    max_unknown_streak: int = 3,
    start_states: int = 1,
    start_reason: str | None = None,
) -> dict:
    """Find s*(n) = smallest s with a SAT verdict, incrementing s from start_states.

    SAT gets easier as s grows, so after the first timeout we keep climbing for a
    few more s to try to pin an upper bound on s*, then give up.

    ``start_states`` implements the monotonicity shortcut: s*(n) is non-decreasing
    in n, because bits 0..n-1 are a prefix of bits 0..n'-1 for n' > n.  So if no
    s-state DFAO fits a shorter prefix, none fits a longer one, and those UNSAT
    verdicts (the expensive ones) need not be re-proved.  They are recorded as
    ``UNSAT_IMPLIED`` with the prefix they were proved on.
    """
    per_state: list[dict] = []
    s_star = None
    first_unknown = None
    unknown_streak = 0
    for s in range(1, start_states):
        per_state.append(
            {
                "states": s,
                "status": "UNSAT_IMPLIED",
                "implied_by": start_reason,
                "solve_s": 0.0,
            }
        )
    for s in range(start_states, max_states + 1):
        res = solve_dfao_sat(bits, states=s, base=base, direction=direction, timeout_s=timeout_s)
        per_state.append(
            {
                "states": s,
                "status": res["status"],
                "solve_s": res["solve_s"],
                "variables": res["variables"],
                "clauses": res["clauses"],
            }
        )
        log(
            f"    {label} s={s:>2} -> {res['status']:<7} "
            f"({res['solve_s']:.2f}s, {res['clauses']} clauses)"
        )
        if res["status"] == "UNKNOWN":
            if first_unknown is None:
                first_unknown = s
            unknown_streak += 1
            if unknown_streak >= max_unknown_streak:
                log(f"    {label} giving up after {unknown_streak} consecutive timeouts")
                break
        else:
            unknown_streak = 0
        if res["status"] == "SAT":
            s_star = s
            per_state[-1]["candidate"] = res["candidate"]
            per_state[-1]["candidate_verified"] = res["candidate_verified"]
            break
    exact = s_star is not None and first_unknown is None
    return {
        "s_star": s_star if exact else None,
        "s_star_exact": exact,
        # every s below first_unknown came back UNSAT, so s* is at least that
        "s_star_lower_bound": first_unknown if first_unknown is not None else s_star,
        "s_star_upper_bound": s_star,
        "search_exhausted_to_states": per_state[-1]["states"] if per_state else 0,
        "hit_timeout": first_unknown is not None,
        "search_started_at_states": start_states,
        "monotonicity_shortcut": start_reason,
        "per_state": per_state,
    }


def run_curve(
    *,
    sequences: list[tuple[str, int]],
    directions: list[str],
    n_values: list[int],
    base: int,
    max_states: int,
    timeout_s: float,
    max_unknown_streak: int = 3,
) -> list[dict]:
    rows: list[dict] = []
    for sequence, seed in sequences:
        for direction in directions:
            log(f"[{sequence} seed={seed} b{base} {direction}]")
            # monotonicity carry: s*(n) is non-decreasing in n
            start_states = 1
            start_reason = None
            for n in sorted(n_values):
                bits = sequence_bits(sequence, n, seed=seed)
                label = f"{sequence}/{direction}/n={n}"
                t0 = time.perf_counter()
                res = min_states_for_prefix(
                    bits,
                    base=base,
                    direction=direction,
                    max_states=max_states,
                    timeout_s=timeout_s,
                    label=label,
                    max_unknown_streak=max_unknown_streak,
                    start_states=start_states,
                    start_reason=start_reason,
                )
                band = counting_null_band(n, base)
                rows.append(
                    {
                        "sequence": sequence,
                        "seed": seed if sequence == "random" else None,
                        "direction": direction,
                        "base": base,
                        "n": n,
                        "sha256_bits_ascii": sha256_ascii(bits_ascii(bits)),
                        "counting_null_band": band,
                        "elapsed_s": round(time.perf_counter() - t0, 6),
                        **res,
                    }
                )
                log(
                    f"  {label}: s* = {res['s_star'] if res['s_star_exact'] else '?'}"
                    f"  (counting null {band['optimistic']}-{band['conservative']})"
                )
                # carry the strongest sound lower bound forward to the next n
                carry = res["s_star_lower_bound"] if res["s_star"] is None else res["s_star"]
                if carry and carry > start_states:
                    start_states = carry
                    start_reason = f"no <{carry}-state DFAO fits the {n}-bit prefix"
    return rows


def summarise(rows: list[dict], n_values: list[int], base: int) -> dict:
    """Pivot the rows into the n | s*(...) | counting-null table."""
    keys = []
    for row in rows:
        key = (row["sequence"], row["direction"])
        if key not in keys:
            keys.append(key)
    table = []
    for n in sorted(n_values):
        band = counting_null_band(n, base)
        entry = {
            "n": n,
            "counting_null_optimistic": band["optimistic"],
            "counting_null_conservative": band["conservative"],
        }
        for sequence, direction in keys:
            match = next(
                (r for r in rows if r["n"] == n and r["sequence"] == sequence and r["direction"] == direction),
                None,
            )
            col = f"{sequence}:{direction}"
            if match is None:
                entry[col] = None
            elif match["s_star_exact"]:
                entry[col] = match["s_star"]
            elif match["s_star_upper_bound"] is not None:
                entry[col] = f"<= {match['s_star_upper_bound']} (>= {match['s_star_lower_bound']})"
            else:
                entry[col] = f"> {match['search_exhausted_to_states']} (unknown)"
        table.append(entry)
    return {"columns": [f"{s}:{d}" for s, d in keys], "rows": table}


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal-DFAO-size curve s*(n)")
    ap.add_argument("--solve-cnf", help=argparse.SUPPRESS)
    ap.add_argument("--solve-result", help=argparse.SUPPRESS)
    ap.add_argument("--bits", default="8,12,16,20,24,28,32,40,48,56,64",
                    help="comma list of prefix lengths n (index 0 included)")
    ap.add_argument("--sequences", default="center,random,thue-morse")
    ap.add_argument("--directions", default="msd,lsd")
    ap.add_argument("--base", type=int, default=2)
    ap.add_argument("--max-states", type=int, default=16)
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--max-unknown-streak", type=int, default=3,
                    help="give up on a prefix after this many consecutive timeouts")
    ap.add_argument("--seed", type=int, default=30, help="seed for the random control")
    ap.add_argument("--gate-only", action="store_true", help="run the correctness gate and stop")
    ap.add_argument("--skip-gate", action="store_true", help="skip the correctness gate (not recommended)")
    ap.add_argument("--out", help="JSON artifact path")
    args = ap.parse_args()

    if not PYSAT_AVAILABLE:
        raise SystemExit(PYSAT_HINT)

    if args.solve_cnf:
        raise SystemExit(solve_cnf_worker(args.solve_cnf, args.solve_result))

    t_start = time.perf_counter()
    n_values = [int(x) for x in args.bits.split(",") if x.strip()]
    directions = [d.strip() for d in args.directions.split(",") if d.strip()]
    sequences = [
        (name.strip(), args.seed if name.strip() == "random" else 0)
        for name in args.sequences.split(",")
        if name.strip()
    ]

    gate = None
    if not args.skip_gate:
        log("running correctness gate (SAT path vs exhaustive search)...")
        gate = run_correctness_gate(timeout_s=args.timeout_s)
        if not gate["passed"]:
            out = {
                "artifact_type": "rule30.dfao_min_state_curve",
                "artifact_version": ARTIFACT_VERSION,
                "created_at": utc_now_iso(),
                "gate": gate,
                "aborted": "correctness gate failed",
            }
            print(json.dumps(out, indent=2))
            raise SystemExit(1)

    rows: list[dict] = []
    if not args.gate_only:
        rows = run_curve(
            sequences=sequences,
            directions=directions,
            n_values=n_values,
            base=args.base,
            max_states=args.max_states,
            timeout_s=args.timeout_s,
            max_unknown_streak=args.max_unknown_streak,
        )

    out = {
        "artifact_type": "rule30.dfao_min_state_curve",
        "artifact_version": ARTIFACT_VERSION,
        "created_at": utc_now_iso(),
        "command": "python " + " ".join(sys.argv),
        **git_context(),
        "claim_level": "Certificate",
        "question": "s*(n) = smallest number of states of a binary-output DFAO reproducing the first n bits.",
        "params": {
            "n_values": n_values,
            "sequences": [s for s, _ in sequences],
            "random_seed": args.seed,
            "directions": directions,
            "base": args.base,
            "max_states": args.max_states,
            "timeout_s": args.timeout_s,
            "max_unknown_streak": args.max_unknown_streak,
        },
        "solver": {
            "engine": SOLVER_NAME,
            "backend": "python-sat (pysat)",
            "pysat_version": getattr(pysat, "__version__", "unknown"),
            "python": sys.version.split()[0],
            "encoding_source": "prize_lab.dfao_sat_cnf (DIMACS, parsed in-process)",
        },
        "counting_null": {
            "source": "experiments/counting_bound.py",
            "reference": "docs/theory/finite-prefix-counting-bound.md",
            "upper_bound_formula": "log2|M(s,b)| <= s*b*log2(s) + s   (used for vacuity verdicts)",
            "lower_bound_formula": "log2|M(s,b)| >= s*b*log2(s) + s - log2((s-1)!)   (used for fair thresholds)",
            "meaning": "a 'no s-state DFAO fits n bits' result is informative only when log2|M(s,b)| >= n",
            "why_a_band": "the two bounds bracket the true behaviour count, so the null for s*(n) is a band",
            "table": dfao_table(args.base, 24),
            "fair_thresholds": fair_thresholds(args.base, sorted(set(n_values + [64, 128, 256]))),
            "prior_certificates_are_vacuous": [
                {
                    "claim": "first 128 center bits are not generated by any 1-4 state binary DFAO (msd or lsd)",
                    "source": "docs/experiment-logs/2026-06-14-prize-dfao-shortcut-smoke.md",
                    **counting_verdict(4, 128, args.base),
                },
                {
                    "claim": "extended to 1-5 states",
                    "source": "docs/experiment-logs/2026-06-15-prize-frontier.md",
                    **counting_verdict(5, 128, args.base),
                },
                {
                    "claim": "planned extension to states 6-8 at n=128",
                    "source": "docs/experiment-logs/2026-06-15-prize-frontier.md",
                    **counting_verdict(8, 128, args.base),
                },
            ],
        },
        "gate": gate,
        "results": rows,
        "summary": summarise(rows, n_values, args.base) if rows else None,
        "elapsed_s": round(time.perf_counter() - t_start, 6),
    }
    if args.out:
        path = Path(args.out)
        if not path.is_absolute():
            path = ROOT / path
        write_json(path, out)
        out["out_path"] = relative_to_root(path)
        log(f"wrote {relative_to_root(path)}")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

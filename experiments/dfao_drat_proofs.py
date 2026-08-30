#!/usr/bin/env python
"""Machine-checkable DRAT proofs for the s*(n) lower bounds.

What this promotes
------------------
``s*(n)``, the minimal-DFAO-size curve, is graded **Robust observation** for
one reason: its *upper* bounds ship a re-verified witness DFAO (certificate
grade already), but its *lower* bounds are CaDiCaL's UNSAT verdicts taken on
trust. This script replaces that trust with a proof another agent can check
with a tool that shares no code with the solver that produced it.

Two things are certified per lower bound ``s*(n) >= k``:

  1. Every instance ``(sequence, n, direction, s)`` for ``s < k`` is UNSAT, and
  2. each UNSAT verdict carries a DRAT refutation accepted by ``drat-trim``.

The matching *upper* bound ``s*(n) <= k`` is certified in the same artifact by
re-running the recorded witness DFAO against the actual bits with an evaluator
written from the definition, sharing no code with the SAT encoding that found
it. Both directions in one place is what makes the row a Certificate rather
than half of one.

Every instance is solved **directly**, including the ones the original run
discharged by monotonicity as ``UNSAT_IMPLIED``. That shortcut is sound - the
constraints for an ``n``-bit prefix are a subset of those for a longer one, so
UNSAT propagates upward - but it is a hand argument sitting in the trusted
base, and at ``n=48`` msd the entire lower bound rested on it with no direct
UNSAT at all. Re-proving all 207 instances removes the argument from the
trusted base rather than asking a reader to accept it.

Why not pysat's ``with_proof=True``
-----------------------------------
It does not work for this purpose. ``Cadical153(with_proof=True).get_proof()``
returns a proof with no terminating empty clause, and ``drat-trim`` rejects it
(``s NOT VERIFIED``) on instances where the same solver run, invoked as a
standalone binary, produces a proof that verifies. pysat's own source carries
the comment ``# stripping may cause issues here!`` at the point where it
``.strip()``s a *binary* DRAT buffer as though it were text. Emitting proofs
through pysat and recording "DRAT proofs emitted" would have produced a
Certificate resting on refutations that do not check.

So proofs come from a standalone solver binary, which also removes pysat from
the trusted base entirely - the point of the exercise.

What is and is not archived
---------------------------
DRAT proofs for these instances run to tens of megabytes each (63 MB for the
hardest), so they are **not** stored, and the recorded ``proof_sha256`` is
provenance for this run, not a reproducibility anchor: a different solver
version will emit a different, equally valid proof.

What *is* reproducible is the CNF. Each instance records the sha256 of its
DIMACS text, so an independent checker regenerates the instance from the
recorded parameters, confirms the hash, and runs its own solver and its own
DRAT checker. That is the check that requires trusting neither this repo nor
its solver.

Toolchain
---------
Needs a DRAT-emitting SAT solver and a DRAT checker on disk:

    bash tools/build_sat_toolchain.sh      # fetches + builds both into third_party/

or point at your own with --cadical / --drat-trim, or the environment
variables CADICAL / DRAT_TRIM.

Usage
-----
    python experiments/dfao_drat_proofs.py \
        --curve data/prize/2026-08-15-dfao-min-state-curve.json \
        --out data/prize/2026-08-30-dfao-drat-proofs.json

    python experiments/dfao_drat_proofs.py --self-test   # ~10 s, no artifact
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dfao_min_states import sequence_bits, sha256_ascii  # noqa: E402
from prize_lab import (  # noqa: E402
    bits_ascii,
    dfao_sat_cnf,
    git_context,
    relative_to_root,
)

ARTIFACT_TYPE = "rule30.dfao_drat_lower_bound_proofs"
ARTIFACT_VERSION = 1

# Solver exit codes, per the SAT competition convention.
RC_SAT, RC_UNSAT = 10, 20

DEFAULT_TOOL_DIR = ROOT / "third_party"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# toolchain discovery
# ---------------------------------------------------------------------------


def find_tool(explicit: str | None, env_var: str, names: list[str],
              hint: str) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get(env_var):
        candidates.append(Path(os.environ[env_var]))
    for name in names:
        candidates.append(DEFAULT_TOOL_DIR / name)
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    raise SystemExit(
        f"could not find {names[0]}.\n"
        f"  tried: {', '.join(str(c) for c in candidates) or '(nothing)'}\n"
        f"  {hint}")


def tool_version(path: Path, args: list[str]) -> str:
    try:
        proc = subprocess.run([str(path)] + args, capture_output=True,
                              text=True, timeout=30)
        out = (proc.stdout or proc.stderr or "").strip().splitlines()
        return out[0].strip() if out else "unknown"
    except Exception:  # noqa: BLE001 - version is metadata, never fatal
        return "unknown"


# ---------------------------------------------------------------------------
# upper bounds: the witness DFAO, evaluated independently
# ---------------------------------------------------------------------------


def eval_dfao(candidate: dict, index: int) -> int:
    """Output of a DFAO on ``index``, implemented from the definition.

    Deliberately does NOT call ``prize_lab.run_dfao``. The witness was found
    by prize_lab's SAT encoding, so checking it with prize_lab's own evaluator
    shares exactly the code a mis-encoding would live in. Same reason
    ``tools/gen_golden_reference.py`` refuses to share code with ``gpu/``.

    Semantics: write ``index`` in base ``b`` (a single 0 digit when the index
    is zero), read the digits most- or least-significant first per
    ``direction``, follow the transition table from the initial state, and
    emit the output attached to the state you land in.
    """
    base = candidate["base"]
    digits: list[int] = []
    n = index
    if n == 0:
        digits.append(0)
    while n:
        digits.append(n % base)
        n //= base
    digits.reverse()                      # most-significant digit first
    if candidate["direction"] == "lsd":
        digits.reverse()

    state = candidate["initial_state"]
    for digit in digits:
        state = candidate["transitions"][state][digit]
    return candidate["outputs"][state]


def certify_upper_bound(row: dict, bits: list[int]) -> dict:
    """Re-verify the SAT witness for ``s*(n) <= k`` against the actual bits."""
    sat = next((p for p in row["per_state"] if p["status"] == "SAT"), None)
    if sat is None or "candidate" not in sat:
        return {"certified": False, "reason": "no witness DFAO recorded"}

    candidate = sat["candidate"]
    produced = [eval_dfao(candidate, i) for i in range(row["n"])]
    mismatches = [i for i, (a, b) in enumerate(zip(produced, bits)) if a != b]
    return {
        "certified": not mismatches,
        "states": candidate["states"],
        "direction": candidate["direction"],
        "first_mismatch": mismatches[0] if mismatches else None,
        "n_mismatches": len(mismatches),
        "evaluator": "independent (experiments/dfao_drat_proofs.eval_dfao)",
        "original_run_said": sat.get("candidate_verified"),
    }


# ---------------------------------------------------------------------------
# one instance
# ---------------------------------------------------------------------------


def certify_instance(bits: list[int], *, states: int, base: int,
                     direction: str, cadical: Path, drat_trim: Path,
                     solve_timeout_s: float, check_timeout_s: float,
                     workdir: Path) -> dict:
    """Solve one DFAO instance and DRAT-check the refutation if it is UNSAT."""
    dimacs, meta = dfao_sat_cnf(bits, states=states, base=base,
                                direction=direction)
    cnf = workdir / "instance.cnf"
    proof = workdir / "instance.drat"
    cnf.write_text(dimacs, encoding="ascii")

    record = {
        "states": states,
        "variables": meta["variables"],
        "clauses": meta["clauses"],
        "cnf_sha256": sha256_text(dimacs),
    }

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [str(cadical), "--no-binary", str(cnf), str(proof)],
            capture_output=True, text=True, timeout=solve_timeout_s)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        record.update(status="UNKNOWN", solve_s=round(time.perf_counter() - t0, 3),
                      drat="timeout-before-proof", verified=False)
        return record
    record["solve_s"] = round(time.perf_counter() - t0, 3)

    if rc == RC_SAT:
        record.update(status="SAT", drat="n/a (satisfiable)", verified=False)
        return record
    if rc != RC_UNSAT:
        record.update(status=f"ERROR(rc={rc})", drat=proc.stderr[-400:],
                      verified=False)
        return record

    record["status"] = "UNSAT"
    record["proof_bytes"] = proof.stat().st_size
    record["proof_sha256"] = sha256_file(proof)

    t1 = time.perf_counter()
    try:
        check = subprocess.run([str(drat_trim), str(cnf), str(proof)],
                               capture_output=True, text=True,
                               timeout=check_timeout_s)
    except subprocess.TimeoutExpired:
        record.update(drat="checker timeout", verified=False,
                      check_s=round(time.perf_counter() - t1, 3))
        return record
    record["check_s"] = round(time.perf_counter() - t1, 3)

    verified = check.returncode == 0 and "s VERIFIED" in check.stdout
    record["drat"] = "VERIFIED" if verified else "NOT VERIFIED"
    record["verified"] = verified
    if not verified:
        record["checker_output"] = (check.stdout + check.stderr)[-800:]
    return record


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------


def self_test(cadical: Path, drat_trim: Path) -> int:
    """Prove the toolchain has detection power before trusting its verdicts.

    A checker that prints VERIFIED unconditionally is worse than no checker,
    so this asserts both directions. The negative controls matter more than
    the positive one and need care: an instance that is UNSAT by unit
    propagation alone is refuted by drat-trim *from the CNF*, so ANY proof of
    it "verifies" - including an empty one. A truncation control on such an
    instance passes vacuously. ``center n=16 s=4`` needs real search, so
    truncating its proof is a control with teeth.
    """
    checks: list[tuple[str, bool, str]] = []

    with tempfile.TemporaryDirectory() as td:
        work = Path(td)

        # 1. A satisfiable instance is reported SAT, not UNSAT.
        tm = sequence_bits("thue-morse", 64)
        r = certify_instance(tm, states=2, base=2, direction="msd",
                             cadical=cadical, drat_trim=drat_trim,
                             solve_timeout_s=120, check_timeout_s=120,
                             workdir=work)
        checks.append(("thue-morse n=64 s=2 msd is SAT",
                       r["status"] == "SAT", r["status"]))

        # 2. A search-requiring UNSAT instance is refuted, and it verifies.
        bits = sequence_bits("center", 16)
        r = certify_instance(bits, states=4, base=2, direction="msd",
                             cadical=cadical, drat_trim=drat_trim,
                             solve_timeout_s=300, check_timeout_s=300,
                             workdir=work)
        checks.append(("center n=16 s=4 msd is UNSAT",
                       r["status"] == "UNSAT", r["status"]))
        checks.append(("its DRAT proof verifies",
                       r.get("verified") is True, str(r.get("drat"))))

        # 3. The checker rejects a truncated proof of that same instance.
        cnf, proof = work / "instance.cnf", work / "instance.drat"
        lines = proof.read_text().splitlines()
        truncated = work / "truncated.drat"
        truncated.write_text("\n".join(lines[:max(1, len(lines) // 2)]) + "\n0\n")
        chk = subprocess.run([str(drat_trim), str(cnf), str(truncated)],
                             capture_output=True, text=True, timeout=300)
        rejected = not (chk.returncode == 0 and "s VERIFIED" in chk.stdout)
        checks.append(("drat-trim rejects a truncated proof", rejected,
                       "rejected" if rejected else "ACCEPTED A BAD PROOF"))

        # 4. The checker rejects the empty clause claimed from a SATISFIABLE
        #    formula. Nothing can refute a satisfiable CNF, so a checker that
        #    accepts this is broken outright.
        sat_cnf, sat_proof = work / "sat.cnf", work / "sat.drat"
        sat_cnf.write_text("p cnf 2 2\n1 0\n2 0\n")
        sat_proof.write_text("0\n")
        chk = subprocess.run([str(drat_trim), str(sat_cnf), str(sat_proof)],
                             capture_output=True, text=True, timeout=120)
        rejected = not (chk.returncode == 0 and "s VERIFIED" in chk.stdout)
        checks.append(("drat-trim rejects refuting a satisfiable formula",
                       rejected,
                       "rejected" if rejected else "ACCEPTED A BAD PROOF"))

    ok = all(passed for _, passed, _ in checks)
    for name, passed, detail in checks:
        log(f"  [{'PASS' if passed else 'FAIL'}] {name}  ({detail})")
    log(f"self-test {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------


def lower_bound_instances(curve: dict, sequences: list[str] | None) -> list[dict]:
    """Every (row, s) pair whose UNSAT is needed for an exact s*(n).

    For a row with exact ``s* = k``, the lower bound is the statement that
    every ``s < k`` is UNSAT. Rows without an exact s* have no lower bound to
    certify and are skipped.
    """
    out = []
    for row in curve["results"]:
        if not row.get("s_star_exact"):
            continue
        if sequences and row["sequence"] not in sequences:
            continue
        for s in range(1, row["s_star"]):
            out.append({"row": row, "states": s})
    return out


def run(curve_path: Path, out_path: Path | None, *, cadical: Path,
        drat_trim: Path, solve_timeout_s: float, check_timeout_s: float,
        sequences: list[str] | None) -> dict:
    curve_path = curve_path.resolve()
    curve = json.loads(curve_path.read_text(encoding="utf-8"))
    todo = lower_bound_instances(curve, sequences)
    log(f"{len(todo)} lower-bound instances to certify "
        f"from {curve_path.name}")

    started = time.time()
    rows: dict[tuple, dict] = {}
    n_verified = n_unsat = n_bad = 0

    with tempfile.TemporaryDirectory(prefix="dfao_drat_") as td:
        work = Path(td)
        for i, item in enumerate(todo, 1):
            row, states = item["row"], item["states"]
            seq, n, direction = row["sequence"], row["n"], row["direction"]
            seed = row["seed"] if seq == "random" else 0
            bits = sequence_bits(seq, n, seed=seed)

            digest = sha256_ascii(bits_ascii(bits))
            if digest != row["sha256_bits_ascii"]:
                raise SystemExit(
                    f"bit prefix for {seq}/n={n} does not reproduce: "
                    f"{digest} != {row['sha256_bits_ascii']}")

            rec = certify_instance(
                bits, states=states, base=row["base"], direction=direction,
                cadical=cadical, drat_trim=drat_trim,
                solve_timeout_s=solve_timeout_s,
                check_timeout_s=check_timeout_s, workdir=work)

            # Cross-check the regenerated encoding against the original run.
            previous = next((p for p in row["per_state"]
                             if p["states"] == states), None)
            if previous and previous.get("clauses") is not None:
                rec["original_status"] = previous["status"]
                rec["encoding_matches_original_run"] = (
                    previous.get("clauses") == rec["clauses"]
                    and previous.get("variables") == rec["variables"])
            else:
                rec["original_status"] = (previous or {}).get("status")

            key = (seq, seed, direction, n)
            entry = rows.setdefault(key, {
                "sequence": seq, "seed": seed, "direction": direction,
                "base": row["base"], "n": n, "s_star": row["s_star"],
                "sha256_bits_ascii": row["sha256_bits_ascii"],
                "lower_bound_claim": f"s*({n}) >= {row['s_star']}",
                "upper_bound_claim": f"s*({n}) <= {row['s_star']}",
                "upper_bound": certify_upper_bound(row, bits),
                "per_state": [],
            })
            entry["per_state"].append(rec)

            if rec["status"] == "UNSAT":
                n_unsat += 1
                if rec.get("verified"):
                    n_verified += 1
                else:
                    n_bad += 1
            else:
                n_bad += 1

            log(f"  [{i:3d}/{len(todo)}] {seq}/{direction}/n={n} s={states} "
                f"-> {rec['status']:<7} {rec.get('drat', ''):<12} "
                f"{rec.get('solve_s', 0):6.2f}s solve "
                f"{rec.get('check_s', 0):6.2f}s check")

    for entry in rows.values():
        entry["per_state"].sort(key=lambda r: r["states"])
        entry["lower_bound_certified"] = all(
            r["status"] == "UNSAT" and r.get("verified") for r in entry["per_state"])
        entry["s_star_certified"] = (entry["lower_bound_certified"]
                                     and entry["upper_bound"]["certified"])

    ordered = sorted(rows.values(),
                     key=lambda e: (e["sequence"], e["direction"], e["n"]))
    certified_rows = [e for e in ordered if e["lower_bound_certified"]]

    artifact = {
        "artifact_type": ARTIFACT_TYPE,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - started, 3),
        "question": (
            "Is every UNSAT verdict backing an exact s*(n) accompanied by a "
            "DRAT refutation that an independent checker accepts?"),
        "source_curve": {
            "path": relative_to_root(curve_path),
            "sha256": sha256_file(curve_path),
            "created_at": curve.get("created_at"),
            "git_head": curve.get("git_head"),
        },
        "method": {
            "monotonicity_shortcut_used": False,
            "note": (
                "Every s < s*(n) is solved directly on the full n-bit prefix. "
                "The original run discharged most of these as UNSAT_IMPLIED "
                "via monotonicity of s* in n; that argument is sound but sits "
                "in the trusted base, and at n=48 msd the whole lower bound "
                "rested on it. Re-proving them removes it."),
            "proofs_archived": False,
            "proof_hash_meaning": (
                "provenance for this run only. A different solver emits a "
                "different, equally valid proof, so proof_sha256 does not "
                "reproduce. The reproducible object is cnf_sha256."),
            "independent_check": (
                "Regenerate each CNF from (sequence, n, direction, base, "
                "states) with prize_lab.dfao_sat_cnf, confirm cnf_sha256, then "
                "run any DRAT-emitting solver and any DRAT checker."),
        },
        "toolchain": {
            "solver": {"path": cadical.name,
                       "version": tool_version(cadical, ["--version"])},
            "checker": {"path": drat_trim.name,
                        "version": "drat-trim (see tools/build_sat_toolchain.sh)"},
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "params": {
            "solve_timeout_s": solve_timeout_s,
            "check_timeout_s": check_timeout_s,
            "sequences": sequences or "all",
        },
        "summary": {
            "instances": n_unsat + n_bad,
            "unsat": n_unsat,
            "drat_verified": n_verified,
            "not_verified_or_unexpected": n_bad,
            "rows_with_certified_lower_bound": len(certified_rows),
            "rows_with_certified_upper_bound": sum(
                1 for e in ordered if e["upper_bound"]["certified"]),
            "rows_with_s_star_fully_certified": sum(
                1 for e in ordered if e["s_star_certified"]),
            "rows_total": len(ordered),
            "encoding_matches_original_run": all(
                r.get("encoding_matches_original_run", True)
                for e in ordered for r in e["per_state"]),
        },
        "git": git_context(),
        "results": ordered,
    }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, indent=2) + "\n",
                            encoding="utf-8", newline="")
        log(f"wrote {out_path}")
    return artifact


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--curve", type=Path,
                    default=ROOT / "data" / "prize" /
                    "2026-08-15-dfao-min-state-curve.json")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--cadical", default=None,
                    help="path to a DRAT-emitting SAT solver binary")
    ap.add_argument("--drat-trim", default=None, help="path to drat-trim")
    ap.add_argument("--solve-timeout-s", type=float, default=1800.0)
    ap.add_argument("--check-timeout-s", type=float, default=3600.0)
    ap.add_argument("--sequences", default=None,
                    help="comma-separated subset, e.g. 'center'")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    cadical = find_tool(
        args.cadical, "CADICAL", ["cadical"],
        "run: bash tools/build_sat_toolchain.sh")
    drat_trim = find_tool(
        args.drat_trim, "DRAT_TRIM", ["drat-trim"],
        "run: bash tools/build_sat_toolchain.sh")
    log(f"solver  {cadical}  ({tool_version(cadical, ['--version'])})")
    log(f"checker {drat_trim}")

    if args.self_test:
        return self_test(cadical, drat_trim)

    artifact = run(args.curve, args.out, cadical=cadical, drat_trim=drat_trim,
                   solve_timeout_s=args.solve_timeout_s,
                   check_timeout_s=args.check_timeout_s,
                   sequences=args.sequences.split(",") if args.sequences else None)
    s = artifact["summary"]
    log(f"\n{s['drat_verified']}/{s['instances']} instances DRAT-verified; "
        f"{s['rows_with_certified_lower_bound']}/{s['rows_total']} lower and "
        f"{s['rows_with_certified_upper_bound']}/{s['rows_total']} upper "
        f"bounds certified; "
        f"{s['rows_with_s_star_fully_certified']}/{s['rows_total']} values of "
        f"s*(n) fully certified")
    if not args.out:
        print(json.dumps(artifact, indent=2))
    return 0 if s["not_verified_or_unexpected"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

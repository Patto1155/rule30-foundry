"""Verify the exact base-2 MSD DFAO minimum at n=64: s*(64)=15.

Static mode regenerates the 14-state n=64 CNF and checks the artifact's
identity fields against it, then independently evaluates the recorded 15-state
witness against the center prefix. --reprove additionally re-solves the
14-state instance and checks its DRAT refutation from scratch.

The refutation is ~2.3 GB and takes ~350 s to solve and ~600 s to check, so the
proof is not retained; --reprove regenerates it from the recorded CNF hash.
"""
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "experiments"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from experiments.dfao_drat_proofs import certify_instance, eval_dfao, sequence_bits
from prize_lab import dfao_sat_cnf

N = 64
LOWER_STATES = 14
UPPER_STATES = 15


def verify(artifact, *, reprove=False, cadical=None, drat_trim=None):
    if artifact.get("n") != N or artifact.get("direction") != "msd" or artifact.get("base") != 2:
        raise ValueError("wrong artifact scope")
    if artifact.get("s_star") != UPPER_STATES:
        raise ValueError("artifact does not claim s*(64)=15")

    bits = sequence_bits("center", N)
    lower = artifact.get("lower_bound", {})
    dimacs, meta = dfao_sat_cnf(bits, states=LOWER_STATES, base=2, direction="msd")
    expected_hash = hashlib.sha256(dimacs.encode("ascii")).hexdigest()
    for key, expected in (("states", LOWER_STATES), ("variables", meta["variables"]),
                          ("clauses", meta["clauses"]), ("cnf_sha256", expected_hash),
                          ("status", "UNSAT"), ("drat", "VERIFIED"), ("verified", True)):
        if lower.get(key) != expected:
            raise ValueError(f"lower-bound field {key} does not match")

    # The upper bound is only as good as an evaluator that shares no code with
    # the encoder that found it, so re-run the witness over the prefix here.
    upper = artifact.get("upper_bound", {})
    if upper.get("states") != UPPER_STATES:
        raise ValueError("upper bound is not the 15-state witness")
    witness = upper.get("witness") or {}
    if (witness.get("states") != UPPER_STATES or witness.get("base") != 2
            or witness.get("direction") != "msd" or witness.get("initial_state") != 0
            or len(witness.get("transitions", [])) != UPPER_STATES
            or len(witness.get("outputs", [])) != UPPER_STATES):
        raise ValueError("invalid witness shape")
    if any(eval_dfao(witness, i) != b for i, b in enumerate(bits)):
        raise ValueError("15-state witness does not reproduce the center prefix")

    # A negative over a class smaller than the prefix is what a coin gives.
    counting = artifact.get("counting_bound", {})
    if counting.get("vacuous") is not False:
        raise ValueError("artifact does not record a non-vacuous counting bound")

    replay = None
    if reprove:
        for name, binary in (("CaDiCaL", cadical), ("drat-trim", drat_trim)):
            if not binary or not Path(binary).is_file():
                raise ValueError(f"{name} binary not found: {binary}")
        with tempfile.TemporaryDirectory() as directory:
            replay = certify_instance(
                bits, states=LOWER_STATES, base=2, direction="msd",
                cadical=Path(cadical).resolve(), drat_trim=Path(drat_trim).resolve(),
                solve_timeout_s=1800, check_timeout_s=3600,
                workdir=Path(directory))
        if (replay.get("cnf_sha256") != expected_hash
                or replay.get("status") != "UNSAT"
                or replay.get("verified") is not True):
            raise ValueError("fresh DRAT replay failed")

    return {"static_checks_verified": True,
            "certificate_reproved": replay is not None,
            "s_star_64_msd": UPPER_STATES,
            "reason": "n=64 s=14 DRAT refutation + independently evaluated s=15 witness",
            "fresh_reproof": replay}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path,
                        default=ROOT / "runs/dfao-n64-msd-s14-2026-09-18.json")
    parser.add_argument("--reprove", action="store_true")
    parser.add_argument("--cadical", default=ROOT / "third_party/cadical")
    parser.add_argument("--drat-trim", default=ROOT / "third_party/drat-trim")
    args = parser.parse_args()
    try:
        report = verify(json.loads(args.artifact.read_text(encoding="utf-8")),
                        reprove=args.reprove, cadical=args.cadical,
                        drat_trim=args.drat_trim)
    except (ValueError, OSError, KeyError) as exc:
        print(f"verify_dfao_n64: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

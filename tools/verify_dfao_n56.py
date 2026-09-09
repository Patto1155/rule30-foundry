"""Verify the targeted base-2 MSD DFAO extension at n=56.

Static mode checks the CNF identity, the previously certified n=48 lower
bound, and the 13-state witness.  --reprove additionally solves the new
12-state n=56 instance and checks its DRAT refutation.
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

from experiments.dfao_drat_proofs import (certify_instance,
                                           certify_upper_bound,
                                           sequence_bits)
from prize_lab import dfao_sat_cnf


def one(rows, **fields):
    matches = [row for row in rows
               if all(row.get(key) == value for key, value in fields.items())]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {fields}, found {len(matches)}")
    return matches[0]


def verify(artifact, curve, anchor, *, reprove=False,
           cadical=None, drat_trim=None):
    if (artifact.get("scope") !=
            "benchmark one n=56 msd 12-state lower-bound instance; not a complete s*(56) certificate"
            or artifact.get("n") != 56 or artifact.get("direction") != "msd"):
        raise ValueError("wrong benchmark scope")
    result = artifact.get("result", {})
    bits = sequence_bits("center", 56)
    dimacs, meta = dfao_sat_cnf(bits, states=12, base=2, direction="msd")
    expected_hash = hashlib.sha256(dimacs.encode("ascii")).hexdigest()
    for key, expected in (("states", 12), ("variables", meta["variables"]),
                          ("clauses", meta["clauses"]),
                          ("cnf_sha256", expected_hash), ("status", "UNSAT"),
                          ("drat", "VERIFIED"), ("verified", True)):
        if result.get(key) != expected:
            raise ValueError(f"benchmark field {key} does not match")

    anchor_row = one(anchor["results"], sequence="center", direction="msd",
                     base=2, n=48)
    if (anchor_row.get("s_star") != 12
            or anchor_row.get("lower_bound_certified") is not True
            or anchor_row.get("s_star_certified") is not True
            or not all(r.get("verified") is True
                       for r in anchor_row.get("per_state", []))):
        raise ValueError("n=48 monotonic lower-bound anchor is not certified")

    curve_row = one(curve["results"], sequence="center", direction="msd",
                    base=2, n=56)
    upper = certify_upper_bound(curve_row, bits)
    if upper.get("certified") is not True or upper.get("states") != 13:
        raise ValueError("13-state upper-bound witness failed")

    replay = None
    if reprove:
        if not cadical or not Path(cadical).is_file():
            raise ValueError(f"CaDiCaL binary not found: {cadical}")
        if not drat_trim or not Path(drat_trim).is_file():
            raise ValueError(f"drat-trim binary not found: {drat_trim}")
        with tempfile.TemporaryDirectory() as directory:
            replay = certify_instance(
                bits, states=12, base=2, direction="msd",
                cadical=Path(cadical).resolve(),
                drat_trim=Path(drat_trim).resolve(),
                solve_timeout_s=600, check_timeout_s=1200,
                workdir=Path(directory))
        if (replay.get("cnf_sha256") != expected_hash
                or replay.get("status") != "UNSAT"
                or replay.get("verified") is not True):
            raise ValueError("fresh DRAT replay failed")
    return {"static_checks_verified": True,
            "certificate_reproved": replay is not None,
            "s_star_56_msd": 13,
            "reason": "certified n=48 lower bound + n=56 s=12 DRAT + s=13 witness",
            "fresh_reproof": replay}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--curve", type=Path,
                        default=ROOT / "data/prize/2026-08-15-dfao-min-state-curve.json")
    parser.add_argument("--anchor", type=Path,
                        default=ROOT / "data/prize/2026-08-30-dfao-drat-proofs.json")
    parser.add_argument("--reprove", action="store_true")
    parser.add_argument("--cadical", default=ROOT / "third_party/cadical")
    parser.add_argument("--drat-trim", default=ROOT / "third_party/drat-trim")
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    result = verify(load(args.artifact), load(args.curve), load(args.anchor),
                    reprove=args.reprove, cadical=args.cadical,
                    drat_trim=args.drat_trim)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

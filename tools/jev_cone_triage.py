#!/usr/bin/env python3
"""Freeze a reproducible allocation queue for diagonal cone-charge refinements.

This prepares research decisions, not a proof search or a theorem certificate.
Unknown exact-enumeration metadata stays null. Baseline/random orders are fixed
before importing measured results; neither is labelled a JEV comparison unless
actual selector decisions and matched solve outcomes have been recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def build_plan(seed: int = 20260919) -> dict:
    """The genuine 75-member bottom/top/count quotient family, frozen a priori."""
    cards = []
    count_bits = {"none": 0, "parity": 2, "exact": 10}
    for bottom, top, count in itertools.product(range(5), range(5), count_bits):
        cards.append({
            "id": f"diag-b{bottom}-t{top}-{count}",
            "bottom_bits_per_diagonal": bottom,
            "top_bits_per_diagonal": top,
            "popcount_feature": count,
            # Ordering cost at q<=24, not an encoding bound: capped endpoint
            # lengths also appear in the actual quotient feature.
            "baseline_feature_cost": 2 * (bottom + top) + count_bits[count],
        })
    cards.sort(key=lambda c: (c["baseline_feature_cost"],
                             c["bottom_bits_per_diagonal"],
                             c["top_bits_per_diagonal"], c["popcount_feature"]))
    baseline = [c["id"] for c in cards]
    shuffled = baseline.copy()
    random.Random(seed).shuffle(shuffled)
    frozen = {
        "candidate_family": "two-diagonal-bottom-top-popcount-quotient",
        "boundary_macrosteps": 12,
        "spatial_block_length": 8,
        "two_diagonal_cut": "q=t+j; retain features of D_(q-1) and D_q",
        "cards": cards,
        "baseline_order": baseline,
        "random_seed": seed,
        "random_order": shuffled,
    }
    return {
        "schema": "jev-cone-triage-v1",
        "goal": "Falsify insufficient finite cut potentials using exact negative cycles.",
        "claim_limit": "A cycle refutes this quotient; absence of sampled cycles proves no all-horizon claim.",
        "exact_metadata_scope": "Counts describe the finite witness-induced graph, not all legal transitions at every depth. Exact-count feature ranges grow with depth.",
        "frozen_design": frozen,
        "frozen_design_sha256": digest(frozen),
        "measurements": {c["id"]: {
            "state_count": None,
            "reachable_transitions": None,
            "negative_cycle": None,
            "rule15_outcome": None,
            "residual_collisions": None,
            "projected_certificate_bytes": None,
            "evidence_source": None,
        } for c in cards},
        "selector_run": {
            "status": "not_run",
            "successful_calls": 0,
            "changed_decisions": None,
            "matched_policy_comparison": None,
        },
    }


def check_plan(plan: dict) -> None:
    frozen = plan["frozen_design"]
    if digest(frozen) != plan["frozen_design_sha256"]:
        raise ValueError("frozen candidate design or policy ordering changed")
    ids = [c["id"] for c in frozen["cards"]]
    if len(ids) != len(set(ids)) or set(plan["measurements"]) != set(ids):
        raise ValueError("duplicate or missing candidate ids")
    for policy in ("baseline_order", "random_order"):
        if sorted(frozen[policy]) != sorted(ids):
            raise ValueError(f"{policy} is not a permutation of all candidates")
    for measurement in plan["measurements"].values():
        if any(value is not None for key, value in measurement.items()
               if key != "evidence_source") and not measurement["evidence_source"]:
            raise ValueError("measured metadata requires an evidence source")


def attach_report(plan: dict, report: dict, source: str, sha256: str) -> None:
    """Import only measured graph fields; leave unperformed controls unknown."""
    check_plan(plan)
    frozen = plan["frozen_design"]
    if (report["macrosteps"] != frozen["boundary_macrosteps"] or
            report["block_space"] != frozen["spatial_block_length"]):
        raise ValueError("report geometry differs from frozen candidate design")
    for result in report["candidates"]:
        spec = result["features"]
        candidate_id = f"diag-b{spec['bottom']}-t{spec['top']}-{spec['count']}"
        if candidate_id not in plan["measurements"]:
            raise ValueError("result candidate was not frozen")
        cycle = result["cycle"]
        if cycle is not None and not result.get("verification", {}).get("verified"):
            raise ValueError("refuse unverified negative-cycle metadata")
        if cycle is not None and (not cycle or sum(e["charge"] for e in cycle) >= 0):
            raise ValueError("reported cycle does not have negative integer charge")
        plan["measurements"][candidate_id].update({
            "state_count": result["states"],
            "reachable_transitions": result["transitions"],
            "negative_cycle": {
                "status": "checked_negative_cycle" if cycle is not None
                          else "none_in_bounded_witness_graph",
                "edge_count": len(cycle) if cycle is not None else None,
                "total_charge": sum(e["charge"] for e in cycle)
                                if cycle is not None else None,
            },
            "evidence_source": source,
        })
    plan["enumeration_evidence"] = {
        "source": source, "sha256": sha256,
        "scope": "Finite witness-induced graph with minimum charge per ordered state pair; not all-depth legal transitions.",
        "global_rule15_control": report["rule15_control"],
    }
    check_plan(plan)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--attach", type=Path,
                        help="Attach an exact enumeration report to existing --out queue")
    args = parser.parse_args()
    if args.attach:
        if not args.out:
            parser.error("--attach requires existing --out queue")
        plan = json.loads(args.out.read_text())
        raw = args.attach.read_bytes()
        source = str(args.attach.resolve().relative_to(Path(__file__).resolve().parents[1]))
        attach_report(plan, json.loads(raw), source, hashlib.sha256(raw).hexdigest())
        args.out.write_text(json.dumps(plan, indent=2, sort_keys=True,
                                       allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"attached_candidates": len(json.loads(raw)["candidates"])}))
        return 0
    if args.check:
        plan = json.loads(args.check.read_text())
        check_plan(plan)
        print(json.dumps({"valid": True,
                          "candidates": len(plan["frozen_design"]["cards"]),
                          "successful_calls": plan["selector_run"]["successful_calls"]}))
        return 0
    if not args.out:
        parser.error("provide --out or --check")
    plan = build_plan()
    check_plan(plan)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Refuse overwriting a freeze with different design or later measurements.
    with args.out.open("x", encoding="utf-8", newline="") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"candidates": len(plan["frozen_design"]["cards"]),
                      "frozen_design_sha256": plan["frozen_design_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

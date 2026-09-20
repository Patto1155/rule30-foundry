#!/usr/bin/env python3
"""Falsify finite-feature potentials on exact growing slanted Rule 30 cuts.

The full cut at depth q is (D_(q-1),D_q), where
D_q(t)=a(t,-q+t), 0<=t<=q.  It grows with q: no bounded-width
closure or forgotten temporal endpoint is asserted here.  Every sampled edge
comes from a genuine finite left-latch cone, over an 8-space/4-macrotime
block.  Quotienting full cuts by chosen features yields necessary difference
constraints for any potential depending only on those features.  An exact
negative cycle disproves that potential family.  Absence of a cycle proves
nothing about unsampled transitions or the cone-charge conjecture.

Cycle witnesses carry full initial rows and are checked by a separately
implemented forward truth-table simulator, not the sideways enumerator.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, asdict
import itertools
import json
import random
from pathlib import Path


@dataclass(frozen=True)
class FeatureSpec:
    bottom: int = 0
    top: int = 0
    count: str = "none"

    def __post_init__(self):
        if self.bottom < 0 or self.top < 0 or self.count not in {"none", "parity", "exact"}:
            raise ValueError("invalid feature specification")


def diagonal_cone(s: int, macrosteps: int, rule: int = 30) -> list[tuple[int, ...]]:
    """Exact full diagonals, each listed from initial time to centre boundary."""
    if macrosteps < 1 or not 0 <= s < 1 << macrosteps or rule not in (15, 30):
        raise ValueError("unsupported cone parameters")
    diagonals = [(0,)]
    for q in range(1, 2 * macrosteps + 1):
        values = [0] * (q + 1)
        values[q] = q & 1
        if rule == 15:
            values[q - 1] = (q - 1) & 1
        else:
            values[q - 1] = 1 if q % 2 == 0 else 1 ^ ((s >> (q // 2)) & 1)
        for t in range(q - 2, -1, -1):
            if rule == 30:
                values[t] = values[t + 1] ^ (diagonals[q - 1][t] | diagonals[q - 2][t])
            else:
                values[t] = 1 ^ values[t + 1]
        diagonals.append(tuple(values))
    return diagonals


def pack(bits) -> int:
    return sum(bit << index for index, bit in enumerate(bits))


def feature(cut: tuple[tuple[int, ...], tuple[int, ...]], spec: FeatureSpec) -> tuple:
    """Retain both characteristic endpoints; lengths saturate at retained bits."""
    result = []
    for word in cut:
        result.extend((min(len(word), spec.bottom), pack(word[:spec.bottom])))
        result.extend((min(len(word), spec.top), pack(word[-spec.top:]) if spec.top else 0))
        if spec.count != "none":
            result.append(sum(word) if spec.count == "exact" else sum(word) % 2)
    return tuple(result)


def full_cut(diagonals, q):
    return (diagonals[q - 1] if q else (), diagonals[q])


def genuine_edges(macrosteps: int = 12, block: int = 8, rule: int = 30):
    """Yield exact bounded witnesses, never an exhaustive all-depth relation."""
    if macrosteps < 1 or macrosteps > 16 or block < 1 or block % 2:
        raise ValueError("use 1..16 macrosteps and positive even block length")
    for s in range(1 << macrosteps):
        diagonals = diagonal_cone(s, macrosteps, rule)
        for q in range(0, 2 * macrosteps - block + 1, block):
            end = q + block
            bits = tuple(diagonals[j][0] for j in range(1, end + 1))
            yield full_cut(diagonals, q), full_cut(diagonals, end), {
                "start_depth": q,
                "end_depth": end,
                "initial_left": pack(bits),
                "charge": 8 * sum(bits[q:end]) - block,
                "rule": rule,
            }


def quotient_graph(witnesses, spec: FeatureSpec):
    nodes = {}
    best = {}
    for source, target, witness in witnesses:
        fs, ft = feature(source, spec), feature(target, spec)
        i = nodes.setdefault(fs, len(nodes))
        j = nodes.setdefault(ft, len(nodes))
        key = (i, j)
        if key not in best or witness["charge"] < best[key]["charge"]:
            best[key] = dict(witness, source=i, target=j)
    return list(nodes), list(best.values())


def negative_cycle(node_count: int, edges: list[dict]) -> list[dict] | None:
    """Exact Bellman-Ford differences, with a shortest-possible self-loop first.

    The queue accelerates relaxations; predecessor depth detects a cycle.
    Returned edges are independently checked before being accepted.
    """
    loops = [edge for edge in edges if edge["source"] == edge["target"] and edge["charge"] < 0]
    if loops:
        return [min(loops, key=lambda edge: (edge["charge"], edge["end_depth"], edge["initial_left"]))]
    outgoing = [[] for _ in range(node_count)]
    for edge in edges:
        outgoing[edge["source"]].append(edge)
    distance = [0] * node_count
    predecessor = [None] * node_count
    depth = [0] * node_count
    pending = deque(range(node_count))
    queued = [True] * node_count
    while pending:
        source = pending.popleft()
        queued[source] = False
        for edge in outgoing[source]:
            target = edge["target"]
            candidate = distance[source] + edge["charge"]
            if candidate >= distance[target]:
                continue
            distance[target] = candidate
            predecessor[target] = edge
            depth[target] = depth[source] + 1
            if depth[target] >= node_count:
                current = target
                for _ in range(node_count):
                    current = predecessor[current]["source"]
                start = current
                cycle = []
                while True:
                    edge_back = predecessor[current]
                    cycle.append(edge_back)
                    current = edge_back["source"]
                    if current == start:
                        break
                cycle.reverse()
                if sum(item["charge"] for item in cycle) >= 0:
                    raise RuntimeError("invalid Bellman-Ford predecessor cycle")
                return cycle
            if not queued[target]:
                pending.append(target)
                queued[target] = True
    return None


def forward_cut(initial: int, depth: int, rule: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Independent cell-by-cell truth-table reconstruction of a full cut.

    Evaluate only the shrinking dependency triangle, avoiding an invented
    zero boundary for nonquiescent Rule 15.
    """
    current = [(initial >> index) & 1 for index in range(depth)]
    first, second = [], []
    for time in range(depth):
        second.append(current[depth - time - 1])
        if time < depth - 1:
            first.append(current[depth - time - 2])
        if (time & 1) and current[0] != 1 and rule == 30:
            raise ValueError("initial prefix fails an alternating Rule 30 latch")
        next_row = []
        for index in range(len(current) - 1):
            left = current[index + 1]
            centre = current[index]
            right = (time & 1) if index == 0 else current[index - 1]
            next_row.append((rule >> (4 * left + 2 * centre + right)) & 1)
        current = next_row
    second.append(depth & 1)
    if depth:
        first.append((depth - 1) & 1)
    return tuple(first), tuple(second)


def verify_cycle(certificate: dict) -> dict:
    """Verify all witnesses and cancellations without sideways enumeration."""
    spec = FeatureSpec(**certificate["features"])
    cycle = certificate["cycle"]
    declared_rule = certificate["rule"]
    block = certificate["block_space"]
    if declared_rule not in (15, 30) or block != 8:
        raise ValueError("unsupported declared certificate domain")
    if not cycle:
        raise ValueError("empty cycle")
    endpoints = []
    total = 0
    seam_differences = []
    full_endpoints = []
    for edge in cycle:
        q, end, initial, rule = (edge[key] for key in ("start_depth", "end_depth", "initial_left", "rule"))
        if (q < 0 or q % block or end != q + block or initial < 0
                or initial >= 1 << end or rule != declared_rule):
            raise ValueError("invalid witness dimensions")
        source = forward_cut(initial & ((1 << q) - 1), q, rule)
        target = forward_cut(initial, end, rule)
        charge = 8 * ((initial >> q).bit_count()) - (end - q)
        if charge != edge["charge"]:
            raise ValueError("incorrect edge charge")
        endpoints.append((feature(source, spec), feature(target, spec)))
        full_endpoints.append((source, target))
        total += charge
    for index, (_, target) in enumerate(endpoints):
        successor = (index + 1) % len(endpoints)
        if target != endpoints[successor][0]:
            raise ValueError("cycle does not close in its declared features")
        actual = full_endpoints[index][1]
        following = full_endpoints[successor][0]
        seam_differences.append({
            "edge": index,
            "target_diagonal_lengths": [len(x) for x in actual],
            "next_source_diagonal_lengths": [len(x) for x in following],
            "full_cuts_equal": actual == following,
            "differing_overlap_indices": [[j for j, (x, y) in enumerate(zip(a, b)) if x != y]
                                          for a, b in zip(actual, following)],
            "target_full_diagonals": [pack(x) for x in actual],
            "next_source_full_diagonals": [pack(x) for x in following],
        })
    if total >= 0:
        raise ValueError("cycle is not negative")
    return {"verified": True, "cycle_edges": len(cycle), "cycle_charge": total,
            "full_cut_seams": seam_differences,
            "conclusion": "No potential of the declared finite features satisfies all genuine sampled edges. This check alone does not decide whether quotient edges concatenate into an infinite cone."}



def verify_report(report: dict) -> dict:
    """Check every negative-cycle certificate carried by a saved report."""
    if "candidates" not in report:
        return verify_cycle(report)
    checked = [verify_cycle(candidate) for candidate in report["candidates"]
               if candidate.get("cycle")]
    return {"verified_negative_cycles": len(checked),
            "cycle_charges": [item["cycle_charge"] for item in checked],
            "claim_limit": "Only supplied negative-cycle witnesses checked; no all-depth closure or no-cycle claim verified."}

def run(macrosteps: int = 12, max_endpoint: int = 3) -> dict:
    witnesses = list(genuine_edges(macrosteps))
    specs = sorted((FeatureSpec(bottom, top, count) for bottom, top, count in
                    itertools.product(range(max_endpoint + 1), range(max_endpoint + 1), ("none", "parity", "exact"))),
                   key=lambda s: (2 * (s.bottom + s.top) + {"none": 0, "parity": 2, "exact": 2 * (2 * macrosteps + 2).bit_length()}[s.count], s.bottom, s.top, s.count))
    reports = []
    for spec in specs:
        nodes, edges = quotient_graph(witnesses, spec)
        cycle = negative_cycle(len(nodes), edges)
        report = {"features": asdict(spec), "rule": 30, "block_space": 8,
                  "feature_domain": "unbounded exact counts" if spec.count == "exact" else "finite",
                  "states": len(nodes), "transitions": len(edges), "cycle": cycle}
        if cycle:
            report["verification"] = verify_cycle(report)
        reports.append(report)
    control_nodes, control_edges = quotient_graph(list(genuine_edges(8, rule=15)), FeatureSpec())
    control = {"features": asdict(FeatureSpec()), "rule": 15, "block_space": 8,
               "cycle": negative_cycle(len(control_nodes), control_edges)}
    return {"experiment": "slanted-cone-potential-falsifier", "macrosteps": macrosteps,
            "block_space": 8, "block_macrotime": 4, "genuine_witnesses": len(witnesses),
            "candidate_order": "deterministic feature-size baseline; no JEV ranking used",
            "candidates": reports, "rule15_control": verify_cycle(control),
            "nonpositive_population_obstruction": population_obstruction(),
            "claim_limit": "Bounded genuine-edge samples can falsify a feature potential via a negative quotient cycle; no-cycle outcomes cannot certify any all-depth inequality."}



def population_obstruction() -> dict:
    """One genuine edge refutes all nonpositive linear population densities."""
    source, target = forward_cut(129, 8, 30), forward_cut(129, 16, 30)
    return {"initial_left": 129, "start_depth": 8, "end_depth": 16,
            "charge": -8, "source_populations": [sum(x) for x in source],
            "target_populations": [sum(x) for x in target],
            "conclusion": "For Phi=-alpha*wt(Dprev)-beta*wt(Dcurr), alpha,beta>=0, the local inequality is -8-5alpha-beta>=0, impossible. Equal-phase constants cancel."}


def sample_integer_potential(node_count: int, edges: list[dict]) -> list[int]:
    """Integer potential on a finite sampled graph, never an all-depth proof."""
    distance = [0] * node_count
    for iteration in range(node_count):
        changed = False
        for edge in edges:
            source, target = edge["source"], edge["target"]
            if distance[target] > distance[source] + edge["charge"]:
                distance[target] = distance[source] + edge["charge"]
                changed = True
        if not changed:
            potential = [-value for value in distance]
            assert all(edge["charge"] + potential[edge["target"]] - potential[edge["source"]] >= 0
                       for edge in edges)
            return potential
    raise ValueError("sampled graph has a negative cycle")


def targeted_refinement() -> dict:
    """Freeze sample potentials, then seek exact violating holdout edges.

    This is a candidate-specific falsification, not an enlarged survival scan.
    The holdout depth32 lies below the previously scanned weight horizon44.
    After each violation the finite sampled potential is repaired. Stop on a
    negative cycle, or after ten repairs / 20,000 draws per frozen potential.
    """
    baseline = list(genuine_edges(12))
    results = []
    for bottom in (3, 4):
        spec = FeatureSpec(bottom=bottom)
        witnesses = baseline[:]
        rng = random.Random(20260919)
        attempts = []
        cycle = None
        for iteration in range(10):
            nodes, edges = quotient_graph(witnesses, spec)
            cycle = negative_cycle(len(nodes), edges)
            if cycle:
                break
            potential = dict(zip(nodes, sample_integer_potential(len(nodes), edges)))
            attempt = {"iteration": iteration, "sample_states": len(nodes),
                       "sample_transitions": len(edges),
                       "frozen_potential": [{"state": state, "value": value}
                                            for state, value in potential.items()]}
            for draw in range(20000):
                s = rng.randrange(1 << 16)
                diagonals = diagonal_cone(s, 16)
                source, target = full_cut(diagonals, 24), full_cut(diagonals, 32)
                fs, ft = feature(source, spec), feature(target, spec)
                charge = 8 * sum(word[0] for word in diagonals[25:]) - 8
                if fs not in potential or ft not in potential:
                    continue
                slack = charge + potential[ft] - potential[fs]
                if slack >= 0:
                    continue
                initial = pack([word[0] for word in diagonals[1:]])
                assert source == forward_cut(initial, 24, 30)
                assert target == forward_cut(initial, 32, 30)
                witness = {"start_depth": 24, "end_depth": 32,
                           "initial_left": initial, "charge": charge, "rule": 30}
                attempt.update({"first_violating_draw_zero_based": draw,
                                "boundary_word": s, "witness": witness,
                                "source_potential": potential[fs], "target_potential": potential[ft],
                                "slack": slack})
                witnesses.append((source, target, witness))
                break
            else:
                attempt["no_violation_within_draw_budget"] = True
                attempts.append(attempt)
                break
            attempts.append(attempt)
        report = {"features": asdict(spec), "rule": 30, "block_space": 8,
                  "attempts": attempts, "cycle": cycle}
        if cycle:
            report["verification"] = verify_cycle(report)
        results.append(report)
    return {"experiment": "targeted-slanted-potential-holdout-falsifier",
            "seed": 20260919, "training_depth": 24, "holdout_block": [24, 32],
            "candidates": results,
            "claim_limit": "Each frozen potential is checked against exact holdout edges. Verified negative quotient cycles refute only the declared feature-potential families."}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macrosteps", type=int, default=12)
    parser.add_argument("--max-endpoint", type=int, default=3)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--targeted-refinement", action="store_true")
    args = parser.parse_args()
    result = (verify_report(json.loads(args.verify.read_text())) if args.verify else
              targeted_refinement() if args.targeted_refinement else
              run(args.macrosteps, args.max_endpoint))
    payload = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8", newline="")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit a recovered B1 artifact without replaying its long map trajectory.

Example: python tools/verify_recovered_b1.py --artifact PATH --self-test
Add --seed-gates to regenerate the finite simulation with the existing map
module. Local consistency is not proof of recorded long trajectory distances
or eternal settlement of the simulation's measured tails.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile


SCOPE = ("Historical artifact consistency and optional finite seed gates only; "
         "excludes long trajectory replay and eternal settlement certification.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def word(value):
    result = int(value, 16)
    require(0 <= result < (1 << 32), "word outside 32 bits")
    return result


def minimal_period(bits):
    return next(q for q in (1, 2, 4, 8, 16, 32, 64)
                if q <= len(bits) and all(
                    bits[t] == bits[(t + q) % len(bits)] for t in range(len(bits))))


def verify(artifact):
    require(artifact["experiment"] == "pattern_map_walk32", "wrong experiment")
    require(artifact["period"] == 32, "expected period 32")
    walk = artifact["walk"]
    branches, events, leaves = walk["branch_points"], walk["events"], walk["exhausted"]
    nodes = branches + events + leaves
    labels = [node["branch"] for node in nodes]
    require(len(labels) == len(set(labels)), "duplicate tree node")
    require("root" in labels, "missing root")
    branch_by_label = {node["branch"]: node for node in branches}
    labels = set(labels)
    for label in labels:
        require(label.startswith("root") and set(label[4:]) <= {"0", "1"},
                "invalid branch label")
        if label != "root":
            require(label[:-1] in branch_by_label, "missing internal parent")
    for branch in branches:
        label = branch["branch"]
        require(label + "0" in labels and label + "1" in labels,
                "missing branch child")
        predecessor = word(branch["predecessor_word"])
        require(predecessor.bit_count() % 2 == 0, "branch parity must be even")
        continuations = [word(w) for w in branch["continuations"]]
        require(len(continuations) == 2 and continuations[0] ^ continuations[1]
                == (1 << 32) - 1, "continuations must be complementary")
        require(continuations[0] & 1 == 0, "branch 0 must start with zero")
        for continuation in continuations:
            require(all(((continuation >> ((t + 1) % 32)) & 1)
                        == (((predecessor >> t) & 1) ^ ((continuation >> t) & 1))
                        for t in range(32)), "invalid cyclic continuation")
    for event in events:
        predecessor = word(event["predecessor_word"])
        bits = [(predecessor >> t) & 1 for t in range(32)]
        require(minimal_period(bits) == 32, "event predecessor not minimal period 32")
        require(predecessor.bit_count() % 2 == 1, "event parity must be odd")
        require(event["minimal_period_of_predecessor"] == 32
                and event["popcount_at_minimal_period"] == predecessor.bit_count(),
                "event period/popcount metadata mismatch")
        require(event["period_before"] == 32 and event["period_after"] == 64,
                "event period metadata mismatch")
        # Check the legacy field as stored, without endorsing its misleading
        # name: these indices are the input (u,0), not the equal-word pair.
        require(event["collision_pair"] == [event["d"] - 2, event["d"] - 1],
                "legacy input-pair indices mismatch")
        x, forced = 0, []
        for t in range(64):
            forced.append(x)
            x ^= bits[t % 32]
        require(x == 0 and minimal_period(forced) == 64, "forced period not 64")
    steps = 0
    horizon = artifact["max_d"]
    require(isinstance(horizon, int) and horizon > artifact["seed"]["d0"],
            "invalid horizon")
    for node in nodes:
        label = node["branch"]
        start = (artifact["seed"]["d0"] if label == "root"
                 else branch_by_label[label[:-1]]["d"] + 1)
        stop = node["d"] if "d" in node else node["reached_d"]
        require(isinstance(stop, int) and start <= stop <= horizon,
                "invalid tree edge distance")
        if "d" in node:
            require(stop < horizon, "event/branch must precede exclusive horizon")
        else:
            require(stop == horizon, "leaf stops before horizon")
        steps += stop - start
    require(steps == walk["steps"], "recorded step count differs from tree sum")
    require(len(events) + len(leaves) == len(branches) + 1, "incomplete binary tree")
    if "conclusion" in artifact:
        conclusion = artifact["conclusion"]
        require(conclusion["doubling_branches"] == len(events)
                and conclusion["undoubled_branches_at_max_d"] == len(leaves),
                "conclusion counts mismatch")
        resolved = bool(events and not leaves)
        require(conclusion["branch_independent_doubling_by_max_d"] == resolved
                and conclusion["actual_diagonal_resolved"] == resolved,
                "conclusion resolution mismatch")
    return {"branch_points_checked": len(branches),
            "cyclic_continuations_checked": 2 * len(branches),
            "event_period64_checks": len(events), "undoubled_leaves": len(leaves),
            "complete_binary_tree_nodes": len(nodes), "recomputed_steps": steps,
            "exclusive_diagonal_horizon": horizon,
            "metadata_note": "collision_pair stores input indices (u,0); equal-word "
                             "collision indices would be (d-3,d-2)."}


def negative_controls(artifact):
    cases = []
    parity = copy.deepcopy(artifact)
    parity["walk"]["events"][0]["predecessor_word"] = hex(
        word(parity["walk"]["events"][0]["predecessor_word"]) ^ 1)
    cases.append(("changed_event_parity", parity))
    continuation = copy.deepcopy(artifact)
    continuation["walk"]["branch_points"][0]["continuations"][0] = hex(
        word(continuation["walk"]["branch_points"][0]["continuations"][0]) ^ 2)
    cases.append(("changed_continuation", continuation))
    steps = copy.deepcopy(artifact)
    steps["walk"]["steps"] += 1
    cases.append(("changed_steps", steps))
    tree = copy.deepcopy(artifact)
    tree["walk"]["exhausted"].pop()
    cases.append(("missing_tree_leaf", tree))
    rejected = []
    for name, changed in cases:
        try:
            verify(changed)
        except ValueError:
            rejected.append(name)
        else:
            raise ValueError(f"negative control accepted: {name}")
    return {"rejected_count": len(rejected), "rejected": rejected}


def seed_gates(artifact):
    # Cache outside the checkout; import the actual existing module unchanged.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import numba
    with tempfile.TemporaryDirectory(prefix="b1-audit-numba-") as cache:
        previous_cache = numba.config.CACHE_DIR
        numba.config.CACHE_DIR = cache
        try:
            from experiments import pattern_map_walk as model
            simulation = artifact["simulation"]
            periods, words, elapsed, base = model.simulate_words(
                simulation["steps"], simulation["diagonals"], simulation["keep"], 32)
            d0 = model.pick_seed(periods, words, 32)
            seed = {"d0": d0, "u": f"0x{int(words[d0-2]):08x}",
                    "v": f"0x{int(words[d0-1]):08x}"}
            require(seed == artifact["seed"], "regenerated seed mismatch")
            histogram = {str(c): int((periods == c).sum())
                         for c in model.np.unique(periods) if c > 0}
            require(histogram == simulation["period_histogram"]
                    and base == simulation["base_row"], "simulation metadata mismatch")
            require(int((periods == 0).sum()) == simulation["unsettled"],
                    "unsettled count mismatch")
            require(int(model.np.flatnonzero(periods == 32)[0])
                    == simulation["first_period32_diagonal"], "first period32 mismatch")
            requested = artifact["gates"]["map_vs_simulation"]["requested"]
            lemma = model.gate_lemma_a(periods, words, d0 - 2, d0 + requested)
            matches = model.gate_map_matches_simulation(words, d0, requested, 32)
            require(lemma == artifact["gates"]["lemma_a"] and lemma["ok"],
                    "lemma gate mismatch")
            require(matches == artifact["gates"]["map_vs_simulation"]
                    and matches["ok"] and matches["checked"] == requested,
                    "map gate mismatch or incomplete coverage")
            return {"seed": seed, "histogram_match": True, "base_match": True,
                    "lemma_a": lemma, "map_vs_simulation": matches,
                    "simulation_seconds": round(elapsed, 3)}
        finally:
            numba.config.CACHE_DIR = previous_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--seed-gates", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        raw = args.artifact.read_bytes()
        artifact = json.loads(raw)
        result = {"status": "pass", "scope": SCOPE,
                  "artifact_sha256": hashlib.sha256(raw).hexdigest(),
                  "consistency": verify(artifact)}
        if args.self_test:
            result["negative_controls"] = negative_controls(artifact)
        if args.seed_gates:
            result["seed_gates"] = seed_gates(artifact)
    except (ValueError, KeyError, TypeError, IndexError, OSError, ImportError) as exc:
        print(json.dumps({"status": "fail", "scope": SCOPE, "error": str(exc)}))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

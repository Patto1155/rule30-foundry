"""Bounded primitive-period sieve for open Rule 30 strips, not seed orbits.

The independent audit uses rule-number lookup and mutual reachability rather
than the search implementation's Boolean update and Kosaraju decomposition.
An implication concerns eventual period p (not necessarily minimal period p).
"""
import argparse
from itertools import product
import json
from pathlib import Path

try:
    from . import periodic_strip_graph as strip
except ImportError:
    import periodic_strip_graph as strip


def necklaces(max_period):
    if type(max_period) is not int or not 1 <= max_period <= 6:
        raise ValueError("max_period must be an integer from 1 to 6")
    words = []
    for length in range(1, max_period + 1):
        for bits in product("01", repeat=length):
            word = "".join(bits)
            if any(length % d == 0 and word == word[:d] * (length // d)
                   for d in range(1, length)):
                continue
            if word == min(word[i:] + word[:i] for i in range(length)):
                words.append(word)
    return words


def validate_widths(widths):
    if (not isinstance(widths, list) or not widths
            or any(type(w) is not int or w not in (3, 5, 7, 9) for w in widths)
            or widths != sorted(set(widths))):
        raise ValueError("widths must be a nonempty increasing subset of 3,5,7,9")


def independent_audit(width, word):
    """Decide the universal recurrent p-step implication independently.

    Every infinite finite-graph walk eventually stays in one recurrent SCC.
    Hence checking every internal p-step path is sufficient. A violating path
    in an SCC closes to a walk and therefore refutes the implication.
    """
    validate_widths([width])
    if not isinstance(word, str) or not word or set(word) - {"0", "1"}:
        raise ValueError("binary word required")
    if len(word) > 6:
        raise ValueError("period exceeds bounded audit")
    radius, period = width // 2, len(word)
    graph = {}
    for phase in range(period):
        for cells in product((0, 1), repeat=width):
            if cells[radius] != int(word[phase]):
                continue
            row = sum(bit << i for i, bit in enumerate(cells))
            middle = [(30 >> (4*cells[i-1] + 2*cells[i] + cells[i+1])) & 1
                      for i in range(1, width-1)]
            next_phase = (phase+1) % period
            successors = []
            if middle[radius-1] == int(word[next_phase]):
                for left, right in product((0, 1), repeat=2):
                    nxt = [left] + middle + [right]
                    successors.append((next_phase, sum(b << i for i, b in enumerate(nxt))))
            graph[phase, row] = successors
    reverse = {v: [] for v in graph}
    for v, edges in graph.items():
        for w in edges:
            reverse[w].append(v)

    def reachable(start, edges, allowed):
        seen, pending = {start}, [start]
        while pending:
            for w in edges[pending.pop()]:
                if w in allowed and w not in seen:
                    seen.add(w)
                    pending.append(w)
        return seen

    remaining, recurrent = set(graph), []
    while remaining:
        pivot = min(remaining)
        group = (reachable(pivot, graph, remaining)
                 & reachable(pivot, reverse, remaining))
        remaining.difference_update(group)
        if len(group) > 1 or pivot in graph[pivot]:
            recurrent.append(group)
    implication = True
    for group in recurrent:
        for start in group:
            frontier = {start}
            for _ in range(period):
                frontier = {w for v in frontier for w in graph[v] if w in group}
            if any(((start[1] ^ end[1]) >> (radius-1)) & 1 for end in frontier):
                implication = False
                break
        if not implication:
            break
    return {"states": len(graph), "recurrent_sccs": len(recurrent),
            "recurrent_states": sum(map(len, recurrent)),
            "classification": ("center_impossible" if not recurrent else
                               "forces_left_period" if implication else
                               "does_not_force_left_period")}


def run(max_period=6, widths=None):
    widths = [3, 5, 7, 9] if widths is None else widths
    validate_widths(widths)
    words = necklaces(max_period)
    records = []
    for word in words:
        for width in widths:
            result = strip.analyze(width, word)
            audit = independent_audit(width, word)
            for key in ("states", "recurrent_sccs", "recurrent_states"):
                if result[key] != audit[key]:
                    raise ValueError("independent graph audit disagrees")
            expected = audit["classification"] != "does_not_force_left_period"
            if result["left_eventually_has_word_period"] != expected:
                raise ValueError("independent implication audit disagrees")
            loops = None
            if not expected:
                loops = strip.output_loops(width, word)
                if loops["found"]:
                    strip.verify_output_loops(loops)
            records.append({"center_word": word, "width": width,
                            "audit": audit, "analysis": result, "output_loops": loops})
    summary = {name: sum(r["audit"]["classification"] == name for r in records)
               for name in ("center_impossible", "forces_left_period",
                            "does_not_force_left_period")}
    summary["nonperiodic_output_loop_pairs"] = sum(
        bool(r["output_loops"] and r["output_loops"]["found"]) for r in records)
    # Normalize tuple states so in-memory and JSON-roundtrip artifacts agree.
    return json.loads(json.dumps({"schema": 1, "max_period": max_period,
                                 "widths": widths, "words": words,
                                 "scope": "open finite strips; no single-seed exclusion claimed",
                                 "summary": summary, "records": records}))


def verify(artifact):
    """Recompute every configured case; reject missing, duplicate or extra data."""
    if not isinstance(artifact, dict) or type(artifact.get("max_period")) is not int:
        raise ValueError("invalid artifact configuration")
    if "widths" not in artifact:
        raise ValueError("missing widths")
    expected = run(artifact["max_period"], artifact["widths"])
    # JSON comparison distinguishes booleans from integers as well as values.
    if json.dumps(artifact, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("artifact differs from complete recomputation")
    return {"verified": True, "records": len(expected["records"]),
            "max_period": expected["max_period"], "widths": expected["widths"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scan", action="store_true")
    mode.add_argument("--verify", type=Path)
    parser.add_argument("--max-period", type=int, default=6)
    parser.add_argument("--widths", type=int, nargs="+", default=[3, 5, 7, 9])
    args = parser.parse_args()
    result = (verify(json.loads(args.verify.read_text(encoding="utf-8")))
              if args.verify else run(args.max_period, args.widths))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Exact finite-strip test of periodic-center -> periodic-left-neighbor.

Open boundaries are free at each step. A counterexample is a finite-strip
closed walk, not a claim that it extends to an infinite-plane seed orbit.
"""
import argparse
from collections import deque
import json
from pathlib import Path


def build_graph(width, word):
    if width < 3 or width % 2 != 1:
        raise ValueError("width must be odd and at least 3")
    if not word or any(c not in "01" for c in word):
        raise ValueError("word must be a nonempty binary string")
    radius, period = width // 2, len(word)
    graph = {}
    for phase in range(period):
        for row in range(1 << width):
            if (row >> radius) & 1 != int(word[phase]):
                continue
            interior = 0
            for i in range(1, width - 1):
                a, b, c = ((row >> j) & 1 for j in (i - 1, i, i + 1))
                interior |= (a ^ (b | c)) << i
            next_phase = (phase + 1) % period
            successors = []
            if (interior >> radius) & 1 == int(word[next_phase]):
                for left in (0, 1):
                    for right in (0, 1):
                        successors.append((next_phase, interior | left | (right << (width - 1))))
            graph[phase, row] = successors
    return graph


def components(graph):
    """Iterative Kosaraju: recursion depth is independent of graph size."""
    seen, order = set(), []
    reverse = {v: [] for v in graph}
    for v, edges in graph.items():
        for w in edges:
            reverse[w].append(v)
    for root in graph:
        if root in seen:
            continue
        seen.add(root)
        stack = [(root, iter(graph[root]))]
        while stack:
            v, edges = stack[-1]
            w = next(edges, None)
            if w is None:
                order.append(v)
                stack.pop()
            elif w not in seen:
                seen.add(w)
                stack.append((w, iter(graph[w])))
    seen.clear()
    result = []
    for root in reversed(order):
        if root in seen:
            continue
        group, stack = set(), [root]
        seen.add(root)
        while stack:
            v = stack.pop()
            group.add(v)
            for w in reverse[v]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        result.append(group)
    return result


def return_path(graph, allowed, start, goal):
    queue, previous = deque([start]), {start: None}
    while queue:
        v = queue.popleft()
        if v == goal:
            path = []
            while v is not None:
                path.append(v)
                v = previous[v]
            return path[::-1]
        for w in graph[v]:
            if w in allowed and w not in previous:
                previous[w] = v
                queue.append(w)
    raise AssertionError("SCC must contain a return path")


def analyze(width, word):
    graph = build_graph(width, word)
    recurrent = [s for s in components(graph)
                 if len(s) > 1 or next(iter(s)) in graph[next(iter(s))]]
    witness = None
    bit_index, period = width // 2 - 1, len(word)
    for group in recurrent:
        for start in sorted(group):
            frontier = {start: [start]}
            for _ in range(period):
                following = {}
                for v, path in frontier.items():
                    for w in graph[v]:
                        if w in group and w not in following:
                            following[w] = path + [w]
                frontier = following
            for end, path in frontier.items():
                if ((start[1] ^ end[1]) >> bit_index) & 1:
                    back = return_path(graph, group, end, start)
                    witness = {"mismatch_path": path, "return_path": back,
                               "closed_walk": path + back[1:]}
                    break
            if witness:
                break
        if witness:
            break
    return {"width": width, "center_word": word,
            "states": len(graph), "recurrent_sccs": len(recurrent),
            "recurrent_states": sum(map(len, recurrent)),
            "left_eventually_has_word_period": witness is None,
            "witness": witness,
            "scope": "open finite strip; no single-seed or infinite-plane counterexample"}


def output_loops(width, word):
    """Find equal-length closed walks with different left-neighbor outputs.

    Synchronized-product BFS is exhaustive within each recurrent SCC. Once
    two equal-length paths meet after an output difference, append the same
    return path. Arbitrary concatenations of the resulting distinct blocks
    include non-eventually-periodic outputs. This remains a strip result.
    """
    graph = build_graph(width, word)
    recurrent = [s for s in components(graph)
                 if len(s) > 1 or next(iter(s)) in graph[next(iter(s))]]
    label_bit = width // 2 - 1
    searched = 0
    for group in recurrent:
        root = min(group)
        initial = (root, root, False)
        queue, previous = deque([initial]), {initial: None}
        goal = None
        while queue:
            pair = queue.popleft()
            a, b, differed = pair
            if differed and a == b:
                goal = pair
                break
            for aa in graph[a]:
                if aa not in group:
                    continue
                for bb in graph[b]:
                    if bb not in group:
                        continue
                    mismatch = bool(((aa[1] ^ bb[1]) >> label_bit) & 1)
                    nxt = (aa, bb, differed or mismatch)
                    if nxt not in previous:
                        previous[nxt] = pair
                        queue.append(nxt)
        searched += len(previous)
        if goal is None:
            continue
        pairs, v = [], goal
        while v is not None:
            pairs.append(v)
            v = previous[v]
        pairs.reverse()
        back = return_path(graph, group, goal[0], root)
        paths = [[pair[j] for pair in pairs] + back[1:] for j in (0, 1)]
        # Label source states: omit the repeated root at each block's end.
        outputs = [[(v[1] >> label_bit) & 1 for v in path[:-1]] for path in paths]
        assert paths[0][0] == paths[1][0] == paths[0][-1] == paths[1][-1]
        assert len(paths[0]) == len(paths[1]) and outputs[0] != outputs[1]
        return {"width": width, "center_word": word, "found": True,
                "product_states_visited": searched,
                "block_length": len(outputs[0]), "output_blocks": outputs,
                "closed_walks": paths,
                "scope": "nonperiodic neighbor possible in open strip; not a seed orbit"}
    return {"width": width, "center_word": word, "found": False,
            "product_states_visited": searched,
            "scope": "no distinct equal-length output loops in recurrent strip SCCs"}


def verify_output_loops(result):
    """Check the supplied witnesses directly, without trusting graph search."""
    width, word = result["width"], result["center_word"]
    if type(width) is not int or width < 3 or width % 2 != 1:
        raise ValueError("invalid width")
    if not word or set(word) - set("01") or result.get("found") is not True:
        raise ValueError("positive binary-word certificate required")
    paths = result["closed_walks"]
    if len(paths) != 2 or len(paths[0]) < 2 or len(paths[0]) != len(paths[1]):
        raise ValueError("two equal positive-length walks required")
    if not paths[0][0] == paths[0][-1] == paths[1][0] == paths[1][-1]:
        raise ValueError("walks must close at the same full state")
    outputs = []
    for path in paths:
        for phase, row in path:
            if (type(phase) is not int or type(row) is not int
                    or not 0 <= phase < len(word) or not 0 <= row < (1 << width)
                    or ((row >> (width//2)) & 1) != int(word[phase])):
                raise ValueError("invalid state or center phase")
        for (phase, row), (next_phase, nxt) in zip(path, path[1:]):
            if next_phase != (phase + 1) % len(word):
                raise ValueError("phase transition mismatch")
            for i in range(1, width-1):
                code = 4*((row >> (i-1)) & 1) + 2*((row >> i) & 1) + ((row >> (i+1)) & 1)
                if ((nxt >> i) & 1) != ((30 >> code) & 1):
                    raise ValueError("interior transition mismatch")
        outputs.append([(v[1] >> (width//2-1)) & 1 for v in path[:-1]])
    if (outputs[0] == outputs[1] or outputs != result["output_blocks"]
            or len(outputs[0]) != result["block_length"]):
        raise ValueError("distinct output blocks or length mismatch")
    return {"width": width, "center_word": word, "verified": True,
            "block_length": len(outputs[0]), "scope": "open finite strip only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scan", action="store_true")
    mode.add_argument("--output-loops", action="store_true")
    mode.add_argument("--verify-loops", type=Path)
    parser.add_argument("--widths", type=int, nargs="+", default=[3, 5, 7, 9])
    parser.add_argument("--words", nargs="+", default=["1", "0", "01"])
    args = parser.parse_args()
    if args.verify_loops:
        certificates = json.loads(args.verify_loops.read_text(encoding="utf-8"))
        print(json.dumps([verify_output_loops(c) for c in certificates], indent=2))
        return
    inspect = output_loops if args.output_loops else analyze
    print(json.dumps([inspect(w, word) for word in args.words
                      for w in args.widths], indent=2))


if __name__ == "__main__":
    main()

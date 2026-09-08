"""Exact finite-strip test of periodic-center -> periodic-left-neighbor.

Open boundaries are free at each step. A counterexample is a finite-strip
closed walk, not a claim that it extends to an infinite-plane seed orbit.
"""
import argparse
from collections import deque
import json


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", action="store_true", required=True)
    parser.add_argument("--widths", type=int, nargs="+", default=[3, 5, 7, 9])
    parser.add_argument("--words", nargs="+", default=["1", "0", "01"])
    args = parser.parse_args()
    print(json.dumps([analyze(w, word) for word in args.words
                      for w in args.widths], indent=2))


if __name__ == "__main__":
    main()

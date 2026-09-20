#!/usr/bin/env python3
"""Exact right-entry basin and shared-predecessor nonentry reduction.

No all-finite-right entry theorem is asserted.  All graphs overapproximate
actual right evolution; extra width is used to disprove spurious recurrence.
"""
from __future__ import annotations

from collections import deque
import json

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

try:
    from .right_coherence_followup import (
        greatest_kernel, independent_return, packed_return, safe_states,
        transition_table,
    )
except ImportError:
    from right_coherence_followup import (
        greatest_kernel, independent_return, packed_return, safe_states,
        transition_table,
    )

WIDTH = 16


def entry_rank(edges, kernel):
    """Least universal basin, with checked strict descent outside K."""
    rank = np.full(len(kernel), -1, dtype=np.int64)
    rank[kernel] = 0
    while True:
        fresh = (rank < 0) & (rank[edges] >= 0).all(axis=0)
        if not fresh.any():
            break
        rank[fresh] = rank[edges[:, fresh]].max(axis=0) + 1
    positive = rank > 0
    assert (rank[edges[:, positive]] < rank[positive]).all()
    assert (rank[edges[:, positive]] >= 0).all()
    assert (rank[edges[:, kernel]] == 0).all()
    return rank


def components(edges, allowed):
    """SCCs of all retained edges, including a separate singleton-loop check."""
    n = len(allowed)
    sources = np.tile(np.arange(n), edges.shape[0])
    targets = edges.ravel()
    good = allowed[sources] & allowed[targets]
    sources, targets = sources[good], targets[good]
    graph = csr_matrix(
        (np.ones(len(sources), dtype=np.int8), (sources, targets)),
        shape=(n, n),
    )
    _, labels = connected_components(graph, connection="strong")
    sizes = np.bincount(labels)
    cyclic_ids = set(np.flatnonzero(sizes > 1).tolist())
    cyclic_ids.update(labels[sources[sources == targets]].tolist())
    cyclic = [np.flatnonzero(labels == i) for i in cyclic_ids]
    return sorted(cyclic, key=len), labels, int((sources == targets).sum())


def output_quotient(edges, states):
    """Refine output bisimulation, then certify any deterministic phase cycle."""
    member = set(map(int, states))
    successors = {
        int(v): sorted({int(w) for w in edges[:, v] if int(w) in member})
        for v in states
    }
    partition = {v: v & 1 for v in successors}
    while True:
        keys = {
            v: (partition[v], tuple(sorted({partition[w] for w in ws})))
            for v, ws in successors.items()
        }
        identifiers = {key: i for i, key in enumerate(sorted(set(keys.values())))}
        refined = {v: identifiers[key] for v, key in keys.items()}
        stable = len(set(partition.values())) == len(identifiers)
        partition = refined
        if stable:
            break
    quotient = {p: set() for p in partition.values()}
    output = {}
    for v, ws in successors.items():
        quotient[partition[v]].update(partition[w] for w in ws)
        output[partition[v]] = v & 1
    period = None
    if all(len(ws) == 1 for ws in quotient.values()):
        start = min(quotient)
        phase, word = {}, []
        v = start
        while v not in phase:
            phase[v] = len(word)
            word.append(str(output[v]))
            v = next(iter(quotient[v]))
        assert v == start and len(phase) == len(quotient)
        for v, ws in quotient.items():
            assert phase[next(iter(ws))] == (phase[v] + 1) % len(phase)
        raw = "".join(word)
        period = min(raw[i:] + raw[:i] for i in range(len(raw)))
    return {
        "states": len(states),
        "output_quotient_classes": len(quotient),
        "largest_quotient_outdegree": max(map(len, quotient.values())),
        "deterministic_output_period_up_to_phase": period,
    }, successors


def lifted_dag_certificate(base_states, width):
    """All width-bit extensions, all true dependency cones, longest DAG path."""
    base = np.asarray(base_states, dtype=np.uint64)
    values = np.sort(np.concatenate([
        base + (np.uint64(a) << np.uint64(WIDTH))
        for a in range(1 << (width - WIDTH))
    ]))
    raw = np.concatenate([
        values + (np.uint64(a) << np.uint64(width)) for a in range(4)
    ])
    direct = independent_return(raw, width)
    packed = packed_return(raw) & np.uint64((1 << width) - 1)
    assert np.array_equal(direct, packed)
    indices = np.searchsorted(values, direct)
    good = indices < len(values)
    good[good] &= values[indices[good]] == direct[good]
    sources = np.tile(np.arange(len(values)), 4)[good]
    targets = indices[good]
    successors = [[] for _ in values]
    indegree = np.zeros(len(values), dtype=np.int64)
    for v, w in zip(sources, targets):
        successors[v].append(int(w))
        indegree[w] += 1
    ready = deque(np.flatnonzero(indegree == 0).tolist())
    rank = np.zeros(len(values), dtype=np.int64)
    visited = 0
    while ready:
        v = ready.popleft()
        visited += 1
        for w in successors[v]:
            rank[w] = max(rank[w], rank[v] + 1)
            indegree[w] -= 1
            if indegree[w] == 0:
                ready.append(w)
    assert visited == len(values), "the lifted graph has a directed cycle"
    assert (rank[targets] > rank[sources]).all()
    return {
        "base_component_size": len(base),
        "width": width,
        "vertices": len(values),
        "independent_truth_table_cones": len(raw),
        "retained_edges_including_multiplicity": len(sources),
        "longest_internal_path_edges": int(rank.max()),
        "maximum_consecutive_observations_in_component": int(rank.max()) + 1,
    }


def outside_core_budget(edges, kernel, recurrent, labels):
    """Weighted condensation rank; actual dwell bounds replace spurious SCCs."""
    core_label = int(labels[recurrent[-1][0]])
    nodes = set(map(int, labels[~kernel])) - {core_label}
    weight = {v: 1 for v in nodes}
    weight[int(labels[recurrent[0][0]])] = 10
    weight[int(labels[recurrent[1][0]])] = 16
    successors = {v: set() for v in nodes}
    indegree = {v: 0 for v in nodes}
    for state in np.flatnonzero(~kernel):
        v = int(labels[state])
        if v not in nodes:
            continue
        for target in edges[:, state]:
            w = int(labels[target])
            if w in nodes and w != v and w not in successors[v]:
                successors[v].add(w)
                indegree[w] += 1
    ready = deque(v for v in nodes if indegree[v] == 0)
    rank = dict(weight)
    visited = 0
    while ready:
        v = ready.popleft()
        visited += 1
        for w in successors[v]:
            rank[w] = max(rank[w], rank[v] + weight[w])
            indegree[w] -= 1
            if indegree[w] == 0:
                ready.append(w)
    assert visited == len(nodes)
    assert all(rank[w] >= rank[v] + weight[w]
               for v in nodes for w in successors[v])
    assert max(rank.values()) == 100
    return {
        "condensation_vertices_outside_K_and_core": len(nodes),
        "maximum_weighted_path_observations": max(rank.values()),
        "meaning": (
            "An actual trajectory never entering K resides in the core forever "
            "from some macrotime at most 100. This does not bound K-entry time."
        ),
    }


def contrasting_core_loops(successors):
    """Find two different output loops; this is only a projected graph witness."""
    base = 4128
    short = [base, 19609, base]
    for v, w in zip(short, short[1:]):
        assert w in successors[v]
    queue = deque([(base, False)])
    predecessor = {(base, False): None}
    endpoint = None
    while queue and endpoint is None:
        v, deviation = queue.popleft()
        for w in successors[v]:
            key = (w, deviation or ((v & 1) == (w & 1)))
            if key in predecessor:
                continue
            predecessor[key] = (v, deviation)
            if w == base and key[1]:
                endpoint = key
                break
            queue.append(key)
    assert endpoint is not None
    long = []
    while endpoint is not None:
        long.append(endpoint[0])
        endpoint = predecessor[endpoint]
    long.reverse()
    for v, w in zip(long, long[1:]):
        assert w in successors[v]
    short_word = "".join(str(v & 1) for v in short[:-1])
    long_word = "".join(str(v & 1) for v in long[:-1])
    assert len(short_word) == 2 and len(long_word) == 37
    assert short_word * 37 != long_word * 2
    return {
        "base": base,
        "short_loop": short,
        "long_loop": long,
        "short_output": short_word,
        "long_output": long_word,
        "equal_length_distinct_output_blocks": 74,
        "scope": "Projected graph only; shared exterior predecessors not supplied.",
    }


def run():
    edges = transition_table(WIDTH)
    kernel, _ = greatest_kernel(edges, safe_states(WIDTH))
    rank = entry_rank(edges, kernel)
    assert int(kernel.sum()) == 13526
    assert int((rank >= 0).sum()) == 21457 and int(rank.max()) == 12
    recurrent, labels, selfloops = components(edges, ~kernel)
    assert [len(v) for v in recurrent] == [9, 29, 2445]
    assert selfloops == 0
    quotients = []
    core_successors = None
    for states in recurrent:
        summary, successors = output_quotient(edges, states)
        quotients.append(summary)
        if len(states) == 2445:
            core_successors = successors
    assert quotients[0]["deterministic_output_period_up_to_phase"] == "0001"
    assert quotients[1]["deterministic_output_period_up_to_phase"] == "0000100001010001"
    small_lifts = [
        lifted_dag_certificate(recurrent[0], 22),
        lifted_dag_certificate(recurrent[1], 18),
    ]
    assert [x["longest_internal_path_edges"] for x in small_lifts] == [9, 15]
    return {
        "width": WIDTH,
        "kernel_size": int(kernel.sum()),
        "universal_entry_basin_size": int((rank >= 0).sum()),
        "maximum_entry_rank_macrosteps": int(rank.max()),
        "cyclic_nonentry_components": quotients,
        "singleton_selfloops": selfloops,
        "shared_predecessor_exclusion_certificates": small_lifts,
        "outside_core_dwell_budget": outside_core_budget(edges, kernel, recurrent, labels),
        "exceptional_core_graph_loop_obstruction": contrasting_core_loops(core_successors),
        "scope": (
            "Every actual finite or infinite right trajectory enters K or eventually "
            "stays in the 2445-state core. No all-finite-right entry theorem, no "
            "claim that the core is physically recurrent or admits aperiodic actual "
            "right traces, and no finite-left growth theorem."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

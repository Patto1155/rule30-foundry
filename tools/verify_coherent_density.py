#!/usr/bin/env python3
"""Independently check the coherent-density integer obstructions.

The checker deliberately does not import the synthesis program.  It rebuilds
each labeled inequality from the formulas in
``docs/theory/coherent-density-derivation.md`` and uses integer arithmetic.
"""

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path


EXPECTED_SCOPES = {
    (False, 8): (66, 168, 9, 2),
    (False, "variable"): (67, 169, 9, 2),
    (True, 8): (122, 276, 24, 6),
    (True, "variable"): (123, 277, 24, 6),
}

TEMPORAL_PAIR_SCOPES = {
    (False, 8): (102, 178, 19),
    (False, "variable"): (103, 179, 19),
    (True, 8): (182, 310, 24),
    (True, "variable"): (183, 311, 24),
}


def rule30(left, centre, right):
    """Return the Rule 30 truth-table entry for three bits."""
    return (30 >> (4 * left + 2 * centre + right)) & 1


def coherent_relation():
    """Enumerate two updates made from one shared seven-cell predecessor."""
    triples = set()
    for u, v, w, z, a, b, c in itertools.product((0, 1), repeat=7):
        r = (a, b, c)
        following = (
            rule30(a, z, w),
            rule30(b, a, z),
            rule30(c, b, a),
        )
        zp = rule30(z, w, v)
        wp = rule30(w, v, u)
        second = (
            rule30(following[0], zp, wp),
            rule30(following[1], following[0], zp),
            rule30(following[2], following[1], following[0]),
        )
        triples.add((r, following, second))
    states = {(r, following) for r, following, _ in triples}
    states.update((following, second) for _, following, second in triples)
    return states, triples


def forward_left_cone(initial_left, depth):
    """Evolve the finite left dependency cone beside the alternating centre.

    Bit zero of ``initial_left`` is ``a(0,1)``, the cell nearest the centre.
    Row ``t`` only retains the cells needed by diagonals through ``depth``.
    """
    if type(initial_left) is not int or initial_left < 0 or depth < 0:
        raise ValueError("the initial left word and depth must be nonnegative")
    rows = [[0] + [(initial_left >> (j - 1)) & 1
                   for j in range(1, depth + 1)]]
    for t in range(depth):
        previous = rows[-1]
        current = [(t + 1) % 2]
        current.extend(
            rule30(previous[j + 1], previous[j], previous[j - 1])
            for j in range(1, depth - t)
        )
        rows.append(current)
    return rows


def _cuts(rows, maximum):
    return {q: tuple(rows[t][q - t] for t in range(q + 1))
            for q in range(maximum + 1)}


COHERENT_STATES, COHERENT_TRIPLES = coherent_relation()
TEMPORAL_STATES = {
    ((r[1], r[2]), (following[1], following[2]))
    for r, following in COHERENT_STATES
}
TEMPORAL_EDGES = {
    (((r[1], r[2]), (following[1], following[2])),
     ((following[1], following[2]), (second[1], second[2])))
    for r, following, second in COHERENT_TRIPLES
}


def _bits(values, description):
    if any(type(value) is not int or value not in (0, 1) for value in values):
        raise ValueError(f"{description} must contain only bits")


def _sequence(value, length, description):
    if type(value) is not list or len(value) != length:
        raise ValueError(f"{description} must be a {length}-bit list")
    _bits(value, description)
    return tuple(value)


def _phase(candidate, tau):
    phases = 2 if candidate["time_parity"] else 1
    if type(tau) is not int or not 0 <= tau < phases:
        raise ValueError("phase is outside the candidate scope")
    return tau


def _density(candidate, p, tau, pair):
    phases = 2 if candidate["time_parity"] else 1
    return ("F", p, tau % phases, *pair)


def _flux(candidate, p, tau, state):
    phases = 2 if candidate["time_parity"] else 1
    return ("psi", p, tau % phases, *state[0], *state[1])


def _terminal(candidate, p, tau, pair):
    phases = 2 if candidate["time_parity"] else 1
    return ("chi", p, tau % phases, *pair)


def _local_density(candidate, p, tau, triple):
    a, b, c = triple
    return [
        (_density(candidate, p, tau, (b, c)), 1),
        (_density(candidate, 1 - p, tau, (a, b)), -1),
    ]


def _top_cap(p, r, following, final_pair):
    """Check the exact alternating-centre cap, independently of graph edges."""
    if p == 0:
        boundary_bit = r[2]
        expected = ((1, 1 - boundary_bit, boundary_bit),
                    (0, boundary_bit, 1 - boundary_bit), (1, 1), 1)
    else:
        u, v = r[0], r[2]
        expected = ((u, 1 - u, v), (1, 1, 1 - v), (0, v), 0)
    if (r, following, final_pair) != expected[:3]:
        raise ValueError("label is not an exact top-boundary row")
    return expected[3]


def reconstruct_constraint(candidate, label):
    """Return ``(integer linear terms, rhs)`` for one documented label."""
    if type(label) is not list or not label or type(label[0]) is not str:
        raise ValueError("constraint label must be a nonempty list")
    kind = label[0]

    if kind == "interior":
        if len(label) != 6:
            raise ValueError("invalid interior label arity")
        _, p, tau, raw_r, raw_following, raw_second = label
        _bits((p,), "parity")
        tau = _phase(candidate, tau)
        r = _sequence(raw_r, 3, "interior row")
        following = _sequence(raw_following, 3, "interior following row")
        second = _sequence(raw_second, 3, "interior second row")
        if (r, following, second) not in COHERENT_TRIPLES:
            raise ValueError("illegal shared-predecessor transition")
        terms = _local_density(candidate, p, tau, r)
        terms += [
            (_flux(candidate, p, tau + 1, (following, second)), 1),
            (_flux(candidate, p, tau, (r, following)), -1),
        ]
        return terms, 0

    if kind == "bottom":
        if len(label) != 4:
            raise ValueError("invalid bottom label arity")
        _, p, raw_r, raw_following = label
        _bits((p,), "parity")
        r = _sequence(raw_r, 3, "bottom row")
        following = _sequence(raw_following, 3, "bottom following row")
        if (r, following) not in COHERENT_STATES:
            raise ValueError("bottom label is not a coherent state")
        a, b, c = r
        terms = [
            (_flux(candidate, p, 0, (r, following)), 1),
            (("g", p, b, c), 1),
            (("g", 1 - p, a, b), -1),
            (("b", p), -1),
        ]
        if candidate["alpha"] == "variable":
            terms.append((("alpha",), c))
            rhs = 1
        else:
            rhs = 1 - candidate["alpha"] * c
        return terms, rhs

    if kind == "top":
        if len(label) != 5:
            raise ValueError("invalid top label arity")
        _, p, raw_r, raw_following, raw_final_pair = label
        _bits((p,), "parity")
        r = _sequence(raw_r, 3, "top row")
        following = _sequence(raw_following, 3, "top following row")
        final_pair = _sequence(raw_final_pair, 2, "top final pair")
        tau = _top_cap(p, r, following, final_pair)
        terms = _local_density(candidate, p, tau, r)
        terms += _local_density(candidate, p, 1 - tau, following)
        terms += [
            (_density(candidate, p, 1 - p, final_pair), 1),
            (_flux(candidate, p, tau, (r, following)), -1),
            (("b", p), 1),
        ]
        return terms, 0

    if kind == "terminal_flow":
        if len(label) != 5:
            raise ValueError("invalid terminal-flow label arity")
        _, p, tau, raw_pair, raw_following = label
        _bits((p,), "parity")
        tau = _phase(candidate, tau)
        pair = _sequence(raw_pair, 2, "terminal pair")
        following = _sequence(raw_following, 2, "terminal following pair")
        b, c = pair
        legal = {
            (rule30(b, a, z), rule30(c, a, b))
            for a, z in itertools.product((0, 1), repeat=2)
        }
        if following not in legal:
            raise ValueError("illegal terminal pair transition")
        return [
            (_density(candidate, p, tau, pair), -1),
            (_terminal(candidate, p, tau, pair), 1),
            (_terminal(candidate, p, tau + 1, following), -1),
        ], 0

    # These rows are absent from the supplied contradictions, but defining
    # them makes the claimed omission check meaningful rather than textual.
    if kind == "initial":
        if len(label) != 2:
            raise ValueError("invalid initialization label arity")
        x = label[1]
        _bits((x,), "initial bit")
        terms = [
            (_density(candidate, 0, 0, (x, 1 - x)), 1),
            (_density(candidate, 0, 1, (1, 1)), 1),
            (("g", 0, x, 1 - x), 1),
        ]
        if candidate["alpha"] == "variable":
            terms.append((("alpha",), 1))
            return terms, 1
        return terms, 1 - candidate["alpha"]

    if kind == "terminal_bound":
        if len(label) != 4:
            raise ValueError("invalid terminal-bound label arity")
        _, p, raw_bottom, raw_top = label
        _bits((p,), "parity")
        bottom = _sequence(raw_bottom, 2, "terminal bottom pair")
        top = _sequence(raw_top, 2, "terminal top pair")
        if (p == 0 and top != (1, 1)) or (p == 1 and top not in ((0, 0), (0, 1))):
            raise ValueError("invalid terminal endpoint")
        return [
            (("g", p, *bottom), -1),
            (_terminal(candidate, p, 0, bottom), -1),
            (_density(candidate, p, 1 - p, top), -1),
            (_terminal(candidate, p, 1 - p, top), 1),
        ], 0

    if kind == "slope_floor":
        if label != ["slope_floor"] or candidate["alpha"] != "variable":
            raise ValueError("invalid slope-floor label")
        return [(("alpha",), 1)], 0

    raise ValueError(f"unknown constraint kind: {kind!r}")


def _scope(candidate):
    parity = candidate.get("time_parity")
    alpha = candidate.get("alpha")
    if type(parity) is not bool or (type(alpha) is not int and alpha != "variable"):
        raise ValueError("unknown candidate scope")
    scope = (parity, alpha)
    if scope not in EXPECTED_SCOPES:
        raise ValueError("unknown candidate scope")
    return scope


def verify_candidate(candidate):
    """Replay one candidate and return its exact contradiction metadata."""
    scope = _scope(candidate)
    variables, inequalities, nonzero_rows, expected_bound = EXPECTED_SCOPES[scope]
    expected_metadata = {
        "variables": variables,
        "inequalities": inequalities,
        "coherent_states": 20,
        "coherent_edges": 44,
        "optimizer_status": 2,
        "outcome": "certificate_language_refuted",
    }
    for key, expected in expected_metadata.items():
        if candidate.get(key) != expected:
            raise ValueError(f"incorrect candidate metadata: {key}")

    entries = candidate.get("contributing_constraints")
    if type(entries) is not list or len(entries) != nonzero_rows:
        raise ValueError("incorrect number of contributing rows")
    seen = set()
    sums = defaultdict(int)
    bound = 0
    kinds = set()
    bottom_emissions = []
    weights = []
    for entry in entries:
        if type(entry) is not dict or set(entry) != {"label", "multiplier"}:
            raise ValueError("invalid contributing-row record")
        multiplier = entry["multiplier"]
        if type(multiplier) is not int or multiplier <= 0:
            raise ValueError("multipliers must be positive integers")
        frozen_label = json.dumps(entry["label"], separators=(",", ":"))
        if frozen_label in seen:
            raise ValueError("duplicate contributing label")
        seen.add(frozen_label)
        terms, rhs = reconstruct_constraint(candidate, entry["label"])
        for variable, coefficient in terms:
            sums[variable] += multiplier * coefficient
        bound += multiplier * rhs
        kinds.add(entry["label"][0])
        if entry["label"][0] == "bottom":
            bottom_emissions.append(entry["label"][2][2])
        weights.append(multiplier)

    residual = {variable: coefficient for variable, coefficient in sums.items() if coefficient}
    if residual or bound <= 0:
        raise ValueError("weighted inequalities do not produce 0 >= a positive integer")
    if bound != expected_bound:
        raise ValueError("unexpected positive contradiction bound")

    dense = candidate.get("farkas_multipliers")
    if (type(dense) is not list or len(dense) != inequalities or
            any(type(value) is not int or value < 0 for value in dense)):
        raise ValueError("invalid dense multiplier vector")
    if [value for value in dense if value] != weights:
        raise ValueError("dense and labeled multipliers disagree")

    verification = candidate.get("verification")
    expected_verification = {
        "verified": True,
        "nonzero_rows": nonzero_rows,
        "identity": f"0 >= {bound}",
        "contradiction": bound,
    }
    if verification != expected_verification:
        raise ValueError("verification summary does not match the exact sum")
    return {
        "time_parity": scope[0],
        "alpha": scope[1],
        "nonzero_rows": nonzero_rows,
        "contradiction": bound,
        "constraint_kinds": sorted(kinds),
        "all_bottom_emissions_zero": bool(bottom_emissions) and not any(bottom_emissions),
    }


def verify_report(report):
    if type(report) is not dict or type(report.get("candidates")) is not list:
        raise ValueError("invalid coherent-density report")
    if (len(COHERENT_STATES), len(COHERENT_TRIPLES)) != (20, 44):
        raise AssertionError("independent Rule 30 enumeration changed")
    results = [verify_candidate(candidate) for candidate in report["candidates"]]
    scopes = {(item["time_parity"], item["alpha"]) for item in results}
    if len(results) != 4 or scopes != set(EXPECTED_SCOPES):
        raise ValueError("report does not cover each known scope exactly once")
    omitted = {"initial", "terminal_bound", "slope_floor"}
    if any(omitted & set(item["constraint_kinds"]) for item in results):
        raise ValueError("endpoint or slope row unexpectedly contributes")
    if not all(item["all_bottom_emissions_zero"] for item in results):
        raise ValueError("a contradiction depends on a positive-emission bottom row")
    return {
        "verified_candidates": len(results),
        "integer_contradictions": [item["contradiction"] for item in results],
        "coherent_states": len(COHERENT_STATES),
        "coherent_edges": len(COHERENT_TRIPLES),
        "omitted_constraint_kinds": sorted(omitted),
        "all_bottom_emissions_zero": True,
        "finite_alpha_independent": True,
        "endpoint_deficit_independent": True,
        "claim": "Only this sufficient coherent-density certificate language is refuted.",
    }


def verify_stronger_obstruction():
    """Check the genuine-cut obstruction that survives any spatial-psi refinement."""
    initial_left = 129
    rows = forward_left_cone(initial_left, 7)
    cuts = _cuts(rows, 7)

    # These are precisely the centre-one updates used below.  Rule 30 can
    # produce the prescribed next centre zero only when a(t,1)=1.
    latch_times = (1, 3, 5)
    latches = {t: rows[t][1] for t in latch_times}
    if any(value != 1 for value in latches.values()):
        raise AssertionError("n=129 does not satisfy the required finite latches")

    expected_cuts = {
        3: (0, 1, 0, 1),
        4: (0, 0, 1, 1, 0),
        5: (0, 0, 1, 0, 1, 1),
        6: (0, 0, 0, 1, 0, 1, 0),
        7: (0, 0, 0, 1, 0, 1, 0, 1),
    }
    if {q: cuts[q] for q in expected_cuts} != expected_cuts:
        raise AssertionError("unexpected n=129 forward cuts")
    odd_endpoints = {q: cuts[q][q - 1] for q in (1, 3, 5, 7)}
    if odd_endpoints != {1: 1, 3: 0, 5: 1, 7: 0}:
        raise AssertionError("unexpected finite latch endpoint choices")

    multiplicities = {4: 2, 5: 2, 6: 1, 7: 1}
    if any(cuts[q][0] != 0 for q in multiplicities):
        raise AssertionError("the selected spatial rows must have zero emission")
    phi_coefficients = defaultdict(int)
    alpha_coefficient = 0
    lower_bound = 0
    for q, multiplier in multiplicities.items():
        phi_coefficients[q] += multiplier
        phi_coefficients[q - 1] -= multiplier
        alpha_coefficient += multiplier * cuts[q][0]
        lower_bound += multiplier
    phi_coefficients = {q: value for q, value in phi_coefficients.items() if value}
    if phi_coefficients != {3: -2, 5: 1, 7: 1}:
        raise AssertionError("spatial multiplicities do not telescope as claimed")

    density_coefficients = defaultdict(int)
    bottom_coefficients = defaultdict(int)
    for q, multiplier in phi_coefficients.items():
        for t in range(q):
            density_coefficients[("F", q % 2, t % 2,
                                  cuts[q - 1][t], cuts[q][t])] += multiplier
        bottom_coefficients[("g", q % 2, cuts[q - 1][0], cuts[q][0])] += multiplier
    density_coefficients = {key: value for key, value in density_coefficients.items()
                            if value}
    bottom_coefficients = {key: value for key, value in bottom_coefficients.items()
                           if value}
    expected_density = {
        ("F", 1, 0, 0, 0): 1,
        ("F", 1, 1, 0, 0): 2,
        ("F", 1, 0, 1, 1): 1,
        ("F", 1, 1, 1, 0): 1,
        ("F", 1, 0, 0, 1): 1,
    }
    if density_coefficients != expected_density or bottom_coefficients:
        raise AssertionError("the exact pair-density difference is incorrect")

    cycles = (
        (((0, 0), (0, 0)), ((0, 0), (0, 0))),
        (((1, 1), (0, 0)), ((0, 0), (1, 1))),
        (((0, 1), (1, 0)), ((1, 0), (0, 1))),
    )
    cycle_density = defaultdict(int)
    cycle_terminal = defaultdict(int)
    for cycle in cycles:
        for tau, (pair, following) in enumerate(cycle):
            b, c = pair
            legal = {
                (rule30(b, a, z), rule30(c, a, b))
                for a, z in itertools.product((0, 1), repeat=2)
            }
            if following not in legal:
                raise AssertionError("claimed terminal-flow edge is illegal")
            cycle_density[("F", 1, tau, *pair)] += 1
            cycle_terminal[("chi", 1, (tau + 1) % 2, *following)] += 1
            cycle_terminal[("chi", 1, tau, *pair)] -= 1
    cycle_density = {key: value for key, value in cycle_density.items() if value}
    cycle_terminal = {key: value for key, value in cycle_terminal.items() if value}
    if cycle_density != expected_density or cycle_terminal:
        raise AssertionError("terminal cycles do not give the same density expression")

    triple_paths = {}
    for q in multiplicities:
        triple_paths[q] = tuple(
            (cuts[q - 2][t], cuts[q - 1][t], cuts[q][t])
            for t in range(q - 1)
        )
    ordered_density = [
        ("F", 1, 0, 0, 0),
        ("F", 1, 1, 0, 0),
        ("F", 1, 0, 1, 1),
        ("F", 1, 1, 1, 0),
        ("F", 1, 0, 0, 1),
    ]
    return {
        "initial_left": initial_left,
        "cuts": {str(q): "".join(map(str, expected_cuts[q])) for q in expected_cuts},
        "triple_paths": {str(q): ["".join(map(str, state)) for state in path]
                         for q, path in triple_paths.items()},
        "required_latches": {str(t): value for t, value in latches.items()},
        "odd_diagonal_endpoint_bits": {str(q): value
                                        for q, value in odd_endpoints.items()},
        "spatial_multiplicities": {str(q): value
                                   for q, value in multiplicities.items()},
        "phi_coefficients": {str(q): value for q, value in phi_coefficients.items()},
        "alpha_coefficient": alpha_coefficient,
        "spatial_lower_bound": lower_bound,
        "bottom_correction_residual": {},
        "density_coefficients": [
            {"variable": list(key), "coefficient": density_coefficients[key]}
            for key in ordered_density
        ],
        "terminal_cycles": [
            [f"{''.join(map(str, pair))}->{''.join(map(str, following))}"
             for pair, following in cycle]
            for cycle in cycles
        ],
        "terminal_upper_bound": 0,
        "contradiction": "Phi_5 + Phi_7 - 2*Phi_3 is both >= 6 and <= 0",
        "scope": ("Any refinement of spatial psi alone fails when it retains "
                  "this pair density F and the terminal-flow inequalities; "
                  "the check uses finite sound left-latch cones only."),
    }


def _temporal_pair(raw, description):
    if type(raw) is not list or len(raw) != 2:
        raise ValueError(f"{description} must contain two pairs")
    return (_sequence(raw[0], 2, description), _sequence(raw[1], 2, description))


def _tf(candidate, p, tau, temporal_pair):
    phases = 2 if candidate["time_parity"] else 1
    return ("TF", p, tau % phases, *temporal_pair[0], *temporal_pair[1])


def _tg(p, temporal_pair):
    return ("tg", p, *temporal_pair[0], *temporal_pair[1])


def _tchi(candidate, p, tau, temporal_pair):
    phases = 2 if candidate["time_parity"] else 1
    return ("tchi", p, tau % phases, *temporal_pair[0], *temporal_pair[1])


def _temporal_projections(r, following):
    return (((r[1], r[2]), (following[1], following[2])),
            ((r[0], r[1]), (following[0], following[1])))


def _temporal_l(candidate, p, tau, r, following):
    newest, older = _temporal_projections(r, following)
    return [(_tf(candidate, p, tau, newest), 1),
            (_tf(candidate, 1 - p, tau, older), -1)]


def reconstruct_temporal_constraint(candidate, label):
    if type(label) is not list or not label or type(label[0]) is not str:
        raise ValueError("temporal constraint label must be a nonempty list")
    kind = label[0]
    if kind == "interior":
        if len(label) != 6:
            raise ValueError("invalid temporal interior label arity")
        _, p, tau, raw_r, raw_following, raw_second = label
        _bits((p,), "parity"); tau = _phase(candidate, tau)
        r = _sequence(raw_r, 3, "row")
        following = _sequence(raw_following, 3, "following row")
        second = _sequence(raw_second, 3, "second row")
        if (r, following, second) not in COHERENT_TRIPLES:
            raise ValueError("illegal shared-predecessor temporal transition")
        return (_temporal_l(candidate, p, tau, r, following) +
                [(_flux(candidate, p, tau + 1, (following, second)), 1),
                 (_flux(candidate, p, tau, (r, following)), -1)]), 0
    if kind == "bottom":
        if len(label) != 4:
            raise ValueError("invalid temporal bottom label arity")
        _, p, raw_r, raw_following = label
        _bits((p,), "parity")
        r = _sequence(raw_r, 3, "bottom row")
        following = _sequence(raw_following, 3, "bottom following row")
        if (r, following) not in COHERENT_STATES:
            raise ValueError("temporal bottom label is not coherent")
        newest, older = _temporal_projections(r, following)
        terms = [(_flux(candidate, p, 0, (r, following)), 1),
                 (_tg(p, newest), 1), (_tg(1 - p, older), -1),
                 (("b", p), -1)]
        if candidate["alpha"] == "variable":
            terms.append((("alpha",), r[2]))
            return terms, 1
        return terms, 1 - candidate["alpha"] * r[2]
    if kind == "top":
        if len(label) != 5:
            raise ValueError("invalid temporal top label arity")
        _, p, raw_r, raw_following, raw_final = label
        _bits((p,), "parity")
        r = _sequence(raw_r, 3, "top row")
        following = _sequence(raw_following, 3, "top following row")
        final = _temporal_pair(raw_final, "top temporal pair")
        tau = _top_cap(p, r, following,
                       (1, 1) if p == 0 else (0, r[2]))
        expected_final = (((r[2], 1 - r[2]), (1, 1)) if p == 0 else
                          ((1, 1 - r[2]), (0, r[2])))
        if final != expected_final:
            raise ValueError("label is not an exact temporal-pair top cap")
        final_tau = 0 if p == 0 else 1
        return (_temporal_l(candidate, p, tau, r, following) +
                [(_tf(candidate, p, final_tau, final), 1),
                 (_flux(candidate, p, tau, (r, following)), -1),
                 (("b", p), 1)]), 0
    if kind == "terminal_flow":
        if len(label) != 5:
            raise ValueError("invalid temporal terminal-flow label arity")
        _, p, tau, raw_u, raw_following = label
        _bits((p,), "parity"); tau = _phase(candidate, tau)
        temporal_pair = _temporal_pair(raw_u, "terminal temporal pair")
        following = _temporal_pair(raw_following, "terminal following pair")
        if (temporal_pair, following) not in TEMPORAL_EDGES:
            raise ValueError("illegal coherent projected transition")
        return [(_tf(candidate, p, tau, temporal_pair), -1),
                (_tchi(candidate, p, tau, temporal_pair), 1),
                (_tchi(candidate, p, tau + 1, following), -1)], 0
    if kind == "slope_floor" and label == ["slope_floor"] and candidate["alpha"] == "variable":
        return [(("alpha",), 1)], 0
    raise ValueError(f"unknown temporal constraint kind: {kind!r}")


def verify_temporal_candidate(candidate):
    scope = _scope(candidate)
    if scope not in TEMPORAL_PAIR_SCOPES:
        raise ValueError("unknown temporal-pair scope")
    variables, inequalities, nonzero_rows = TEMPORAL_PAIR_SCOPES[scope]
    metadata = {"variables": variables, "inequalities": inequalities,
                "coherent_states": 20, "coherent_edges": 44,
                "temporal_pair_states": 10, "temporal_pair_edges": 22,
                "optimizer_status": 2, "outcome": "certificate_language_refuted"}
    if any(candidate.get(key) != value for key, value in metadata.items()):
        raise ValueError("incorrect temporal-pair candidate metadata")
    entries = candidate.get("contributing_constraints")
    if type(entries) is not list or len(entries) != nonzero_rows:
        raise ValueError("incorrect temporal-pair contributing row count")
    sums = defaultdict(int); bound = 0; weights = []; kinds = set(); bottoms = []; seen = set()
    for entry in entries:
        if type(entry) is not dict or set(entry) != {"label", "multiplier"}:
            raise ValueError("invalid temporal contributing-row record")
        weight = entry["multiplier"]
        if type(weight) is not int or weight <= 0:
            raise ValueError("temporal multipliers must be positive integers")
        frozen = json.dumps(entry["label"], separators=(",", ":"))
        if frozen in seen:
            raise ValueError("duplicate temporal contributing label")
        seen.add(frozen)
        terms, rhs = reconstruct_temporal_constraint(candidate, entry["label"])
        for variable, coefficient in terms:
            sums[variable] += weight * coefficient
        bound += weight * rhs; weights.append(weight); kinds.add(entry["label"][0])
        if entry["label"][0] == "bottom": bottoms.append(entry["label"][2][2])
    if any(sums.values()) or bound != 4:
        raise ValueError("temporal weighted rows do not give exactly 0 >= 4")
    dense = candidate.get("farkas_multipliers")
    if (type(dense) is not list or len(dense) != inequalities or
            any(type(value) is not int or value < 0 for value in dense) or
            [value for value in dense if value] != weights):
        raise ValueError("invalid temporal dense multiplier vector")
    expected = {"verified": True, "nonzero_rows": nonzero_rows,
                "identity": "0 >= 4", "contradiction": 4}
    if candidate.get("verification") != expected:
        raise ValueError("incorrect temporal verification summary")
    return {"time_parity": scope[0], "alpha": scope[1], "contradiction": 4,
            "constraint_kinds": sorted(kinds),
            "all_bottom_emissions_zero": bool(bottoms) and not any(bottoms)}


def verify_temporal_controls():
    if (len(TEMPORAL_STATES), len(TEMPORAL_EDGES)) != (10, 22):
        raise AssertionError("incorrect coherent temporal projection")
    cuts = _cuts(forward_left_cone(129, 7), 7)
    checked = []
    for q in range(3, 8):
        p = q % 2
        left = defaultdict(int); right = defaultdict(int)
        for t in range(q - 1):
            u = ((cuts[q - 1][t], cuts[q][t]),
                 (cuts[q - 1][t + 1], cuts[q][t + 1]))
            left[("F", p, t % 2, *u[0], *u[1])] += 1
        u0 = ((cuts[q - 1][0], cuts[q][0]),
              (cuts[q - 1][1], cuts[q][1]))
        left[("g", p, *u0[0], *u0[1])] += 1
        for t in range(q - 2):
            r = (cuts[q - 2][t], cuts[q - 1][t], cuts[q][t])
            rp = (cuts[q - 2][t + 1], cuts[q - 1][t + 1], cuts[q][t + 1])
            newest, older = _temporal_projections(r, rp)
            right[("F", p, t % 2, *newest[0], *newest[1])] += 1
            right[("F", 1 - p, t % 2, *older[0], *older[1])] -= 1
        final_t = q - 2
        final = ((cuts[q - 1][final_t], cuts[q][final_t]),
                 (cuts[q - 1][final_t + 1], cuts[q][final_t + 1]))
        top_t = q - 3
        top_r = (cuts[q - 2][top_t], cuts[q - 1][top_t], cuts[q][top_t])
        top_rp = (cuts[q - 2][top_t + 1], cuts[q - 1][top_t + 1], cuts[q][top_t + 1])
        _top_cap(p, top_r, top_rp, (1, 1) if p == 0 else (0, top_r[2]))
        expected_final = (((top_r[2], 1 - top_r[2]), (1, 1)) if p == 0 else
                          ((1, 1 - top_r[2]), (0, top_r[2])))
        if final != expected_final:
            raise AssertionError("actual cut does not match temporal top cap")
        right[("F", p, final_t % 2, *final[0], *final[1])] += 1
        newest, older = _temporal_projections(
            (cuts[q - 2][0], cuts[q - 1][0], cuts[q][0]),
            (cuts[q - 2][1], cuts[q - 1][1], cuts[q][1]))
        right[("g", p, *newest[0], *newest[1])] += 1
        right[("g", 1 - p, *older[0], *older[1])] -= 1
        # Subtract Phi_(q-1) on the left.
        for t in range(q - 2):
            u = ((cuts[q - 2][t], cuts[q - 1][t]),
                 (cuts[q - 2][t + 1], cuts[q - 1][t + 1]))
            left[("F", 1 - p, t % 2, *u[0], *u[1])] -= 1
        old0 = ((cuts[q - 2][0], cuts[q - 1][0]),
                (cuts[q - 2][1], cuts[q - 1][1]))
        left[("g", 1 - p, *old0[0], *old0[1])] -= 1
        if {k:v for k,v in left.items() if v} != {k:v for k,v in right.items() if v}:
            raise AssertionError("temporal spatial telescope failed")
        # Every nonfinal temporal transition is in the 22-edge terminal graph.
        us = [((cuts[q - 1][t], cuts[q][t]),
               (cuts[q - 1][t + 1], cuts[q][t + 1])) for t in range(q - 1)]
        if any((us[t], us[t + 1]) not in TEMPORAL_EDGES for t in range(len(us)-1)):
            raise AssertionError("actual cut leaves projected terminal graph")
        checked.append(q)
    return {"spatial_telescope_depths": checked, "q3_interior_edges": 0,
            "temporal_pair_states": 10, "temporal_pair_edges": 22,
            "terminal_extrema_domain_sizes": {"bottom": 10, "top_even": 2, "top_odd": 4},
            "initialization": "finite minimum over two exact q=2 states",
            "terminal_bound": "finite maxima over the reported finite domains"}


def verify_temporal_report(report):
    if type(report) is not dict or type(report.get("candidates")) is not list:
        raise ValueError("invalid temporal-pair report")
    results = [verify_temporal_candidate(candidate) for candidate in report["candidates"]]
    if (len(results) != 4 or
            {(r["time_parity"], r["alpha"]) for r in results} != set(TEMPORAL_PAIR_SCOPES)):
        raise ValueError("temporal report does not cover each scope once")
    if any("slope_floor" in r["constraint_kinds"] for r in results):
        raise ValueError("temporal witness unexpectedly uses the alpha floor")
    if not all(r["all_bottom_emissions_zero"] for r in results):
        raise ValueError("temporal witness uses a positive-emission bottom")
    return {"verified_candidates": 4, "integer_contradictions": [4, 4, 4, 4],
            "all_alpha_coefficients_cancel": True,
            "initialization_constraints": 0,
            "controls": verify_temporal_controls(),
            "claim": "Only the temporal-pair sufficient language is refuted."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--temporal-pair-certificate", type=Path)
    args = parser.parse_args()
    result = verify_report(json.loads(args.certificate.read_text()))
    result["stronger_obstruction"] = verify_stronger_obstruction()
    if args.temporal_pair_certificate:
        result["temporal_pair"] = verify_temporal_report(
            json.loads(args.temporal_pair_certificate.read_text()))
    print(json.dumps(result))


if __name__ == "__main__":
    main()

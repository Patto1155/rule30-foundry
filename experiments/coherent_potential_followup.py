#!/usr/bin/env python3
"""Exact coefficient-independent endpoint-potential obstructions.

Every cycle edge is a physically coherent finite full-row witness. The cycles
close only after taking endpoint features, not as complete spacetime cuts.
Default verification uses an elementary-rule truth table, independent of the
sideways and packed integer recurrences used by optional witness discovery.
"""
from __future__ import annotations

import argparse
import json
import random


# (start depth, initial left integer, initial right integer), centre initially 0.
# Bit zero in either integer is the neighbour closest to the centre.
SMALL_CYCLE = (
    (98, 16661046055632894622381385253, 2587743454036826996634021153954),
    (133, 560767664687031763042728977378771580502,
     46182698749820787015413023666985299359229),
)
REFINED_CYCLE = (
    (119, 23377037085172303049948036924773925,
     17174799572441196282000698251641293466),
    (140, 36087521634883805972806048435262185809445,
     8985863298854676549430037236045755609289314),
    (133, 532154234490797596253366005610507436353,
     5377943350200483748494005685650859662320),
    (126, 4291919420248050447960120932528241942,
     5684538615578513586148010115811968653551),
)


def pack(bits):
    return sum(bit << index for index, bit in enumerate(bits))


def physical_cuts(q, left, right):
    """Replay one finite physical row, returning cuts at q and q+7.

    The dependency interval starts at [-end,end]. At time t it shrinks to
    [-end+t,end-t]. D_j(t)=cell(t,t-j) therefore has array index end-j.
    No prescribed centre boundary is used during the replay.
    """
    end = q + 7
    if q < 8 or q % 7 or min(left, right) < 0:
        raise ValueError("invalid witness dimensions")
    if left >= 1 << end or right >= 1 << end:
        raise ValueError("witness extends outside declared cone")
    row = [((left >> (-x - 1)) & 1) if x < 0 else
           ((right >> (x - 1)) & 1) if x > 0 else 0
           for x in range(-end, end + 1)]
    diagonals = {j: [] for j in (q - 1, q, end - 1, end)}
    for time in range(end + 1):
        if row[end - time] != time % 2:
            raise ValueError(f"physical centre fails at time {time}")
        for depth in diagonals:
            if time <= depth:
                diagonals[depth].append(row[end - depth])
        row = [(30 >> (4 * row[i - 1] + 2 * row[i] + row[i + 1])) & 1
               for i in range(1, len(row) - 1)]
    return ((tuple(diagonals[q - 1]), tuple(diagonals[q])),
            (tuple(diagonals[end - 1]), tuple(diagonals[end])))


def endpoint_feature(cut, depth, bottom, top):
    """Phase, then bottom/top bits of older and current diagonal, in order."""
    if min(map(len, cut)) < max(bottom, top):
        raise ValueError("feature window exceeds diagonal")
    return (depth % 2, *(value for word in cut for value in
                       (pack(word[:bottom]), pack(word[-top:]))))


def verify_cycle(witnesses, bottom=2, top=2):
    records = []
    full_cuts = []
    for q, left, right in witnesses:
        source, target = physical_cuts(q, left, right)
        weight = (left >> q).bit_count()
        if weight:
            raise ValueError("coefficient-independent witness must have weight zero")
        records.append({
            "start_depth": q, "end_depth": q + 7,
            "initial_left": left, "initial_right": right,
            "slab_initial_weight": weight,
            "source_feature": endpoint_feature(source, q, bottom, top),
            "target_feature": endpoint_feature(target, q + 7, bottom, top),
            "centre_alternates_through_time": q + 7,
        })
        full_cuts.append((source, target))
    seams = []
    for i, record in enumerate(records):
        successor = (i + 1) % len(records)
        if record["target_feature"] != records[successor]["source_feature"]:
            raise ValueError("cycle fails to close in declared features")
        target, next_source = full_cuts[i][1], full_cuts[successor][0]
        seams.append({
            "target_lengths": list(map(len, target)),
            "next_source_lengths": list(map(len, next_source)),
            "full_cuts_equal": target == next_source,
        })
    return {
        "bottom_bits_per_diagonal": bottom, "top_bits_per_diagonal": top,
        "phase_retained": "spatial depth modulo two",
        "verified_edges": records, "seams": seams,
        "total_weight": 0, "total_slab_demand": 7 * len(records),
        "conclusion": "No function of these features satisfies A*w-7 >= Phi(target)-Phi(source) on every physical finite-prefix cut, for any finite A.",
    }


def discover(draws=10000, bottom=2, top=8):
    """Optional deterministic exact-witness search, not a survival survey.

    Target only zero-weight seven-cell edges; stop as soon as their feature
    graph has a cycle. The returned cycle is independently physically replayed.
    """
    try:
        from experiments.alternating_right_coherence import boundary_word
        from experiments.slanted_cone_potential import negative_cycle
    except ModuleNotFoundError:
        from alternating_right_coherence import boundary_word
        from slanted_cone_potential import negative_cycle

    rng = random.Random(123)
    nodes, edges = {}, {}
    for draw in range(draws):
        right = draw if draw < 1000 else rng.getrandbits(160)
        s = int(boundary_word(right, 80)[::-1], 2)
        diagonals = [0]
        for depth in range(1, 161):
            bit = 1 if depth % 2 == 0 else 1 ^ ((s >> (depth // 2)) & 1)
            mask = (1 << (depth - 1)) - 1
            word = (diagonals[-1] | (diagonals[-2] if depth > 1 else 0)) & mask
            shift = 1
            while shift < depth:
                word ^= word >> shift
                shift *= 2
            diagonals.append((word ^ (mask if bit else 0)) |
                             (bit << (depth - 1)) | ((depth % 2) << depth))
        initial = pack(word & 1 for word in diagonals[1:])

        def packed_feature(depth):
            return (depth % 2, *(value for j in (depth - 1, depth) for value in
                   (diagonals[j] & ((1 << bottom) - 1),
                    diagonals[j] >> (j + 1 - top))))

        for q in range(14, 154, 7):
            if (initial >> q) & 127:
                continue
            source = nodes.setdefault(packed_feature(q), len(nodes))
            target = nodes.setdefault(packed_feature(q + 7), len(nodes))
            edges.setdefault((source, target), {
                "source": source, "target": target, "charge": -7,
                "start_depth": q, "initial_left": initial & ((1 << (q + 7)) - 1),
                "initial_right": right & ((1 << (q + 7)) - 1),
            })
        cycle = negative_cycle(len(nodes), list(edges.values()))
        if cycle:
            report = verify_cycle([(edge["start_depth"], edge["initial_left"],
                                    edge["initial_right"]) for edge in cycle], bottom, top)
            return {"first_cycle_draw_zero_based": draw, "certificate": report}
    return {"draws": draws, "no_zero_weight_cycle_found": True,
            "claim_limit": "No-cycle sampled result proves no all-depth inequality."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discover", action="store_true")
    args = parser.parse_args()
    result = discover() if args.discover else {
        "small_cycle": verify_cycle(SMALL_CYCLE),
        "refined_cycle": verify_cycle(REFINED_CYCLE, top=8),
        "claim_limit": "These quotient cycles refute only the declared feature potentials. Their complete cuts differ at seams; no infinite physical alternating orbit or failure of a richer potential is established.",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

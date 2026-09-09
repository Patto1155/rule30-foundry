"""Compare saved open-strip loops with actual single-seed strip rows.

An alignment starts only when a saved loop state exactly equals an actual
Rule 30 row around the center.  We then compare one complete loop block.
This is a bounded compatibility observation, not an asymptotic exclusion.
"""
import argparse
import json
from pathlib import Path

try:
    from .periodic_strip_graph import verify_output_loops
except ImportError:
    from periodic_strip_graph import verify_output_loops


def seed_strip_rows(widths, last_time):
    """Return actual centered strip codes for times 0..last_time."""
    if (not widths or any(type(w) is not int or w < 3 or w % 2 == 0
                          for w in widths)
            or type(last_time) is not int or last_time < 0):
        raise ValueError("need odd widths >=3 and a nonnegative last_time")
    radius = max(widths) // 2
    center = last_time + radius + 2
    tape_width = 2 * center + 1
    mask = (1 << tape_width) - 1
    state = 1 << center
    result = {w: [] for w in widths}
    for _ in range(last_time + 1):
        for w in widths:
            r = w // 2
            result[w].append((state >> (center - r)) & ((1 << w) - 1))
        state = ((state << 1) ^ (state | (state >> 1))) & mask
    return result


def first_mismatch(actual, expected, bit=None):
    """First 1-based mismatch; step zero is the required exact alignment."""
    for k, (a, e) in enumerate(zip(actual[1:], expected[1:]), 1):
        if ((a ^ e) if bit is None else ((a ^ e) >> bit) & 1):
            return k
    return None


def analyze(certificates, max_onset=4096):
    if type(max_onset) is not int or max_onset < 0:
        raise ValueError("max_onset must be a nonnegative integer")
    if not isinstance(certificates, list) or not certificates:
        raise ValueError("nonempty certificate list required")
    for certificate in certificates:
        verify_output_loops(certificate)
    widths = [c["width"] for c in certificates]
    if widths != sorted(set(widths)):
        raise ValueError("certificate widths must be unique and increasing")
    max_block = max(c["block_length"] for c in certificates)
    actual = seed_strip_rows(widths, max_onset + max_block)
    records = []
    for certificate in certificates:
        width = certificate["width"]
        radius = width // 2
        hits = []
        for choice, closed in enumerate(certificate["closed_walks"]):
            cycle = closed[:-1]
            block_length = len(cycle)
            for onset in range(max_onset + 1):
                for offset, (_, root) in enumerate(cycle):
                    if root != actual[width][onset]:
                        continue
                    expected = [cycle[(offset + k) % block_length][1]
                                for k in range(block_length + 1)]
                    observed = actual[width][onset:onset + block_length + 1]
                    center = first_mismatch(observed, expected, radius)
                    left = first_mismatch(observed, expected, radius - 1)
                    full = first_mismatch(observed, expected)
                    hits.append({
                        "onset": onset,
                        "choice": choice,
                        "loop_offset": offset,
                        "first_full_strip_mismatch": full,
                        "first_center_mismatch": center,
                        "first_left_neighbor_mismatch": left,
                        "left_mismatch_precedes_center":
                            left is not None and (center is None or left < center),
                    })
        def matched(field, length):
            return max((h[field] if h[field] is not None else length + 1) - 1
                       for h in hits) if hits else None
        records.append({
            "width": width,
            "block_length": certificate["block_length"],
            "exact_initial_alignments": len(hits),
            "max_full_strip_steps_matched":
                matched("first_full_strip_mismatch", certificate["block_length"]),
            "max_center_steps_matched":
                matched("first_center_mismatch", certificate["block_length"]),
            "max_left_neighbor_steps_matched":
                matched("first_left_neighbor_mismatch", certificate["block_length"]),
            "alignments_with_left_mismatch_before_center":
                sum(h["left_mismatch_precedes_center"] for h in hits),
            "alignments_matching_entire_block":
                sum(h["first_full_strip_mismatch"] is None for h in hits),
            "alignments": hits,
        })
    return {
        "schema": 1,
        "max_onset": max_onset,
        "widths": widths,
        "scope": "one-block actual-seed compatibility; no asymptotic exclusion",
        "records": records,
    }


def verify(artifact, certificates):
    if not isinstance(artifact, dict) or type(artifact.get("max_onset")) is not int:
        raise ValueError("invalid artifact")
    expected = analyze(certificates, artifact["max_onset"])
    if artifact != expected:
        raise ValueError("artifact differs from complete recomputation")
    return {"verified": True,
            "widths": expected["widths"],
            "alignments": sum(r["exact_initial_alignments"]
                              for r in expected["records"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loops", type=Path, required=True)
    parser.add_argument("--max-onset", type=int, default=4096)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    certificates = json.loads(args.loops.read_text(encoding="utf-8"))
    result = (verify(json.loads(args.verify.read_text(encoding="utf-8")),
                     certificates)
              if args.verify else analyze(certificates, args.max_onset))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n",
                            encoding="utf-8", newline="")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

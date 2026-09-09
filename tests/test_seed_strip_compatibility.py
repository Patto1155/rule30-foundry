import copy
import json
import unittest
from pathlib import Path

from experiments import seed_strip_compatibility as probe


ROOT = Path(__file__).resolve().parents[1]
LOOPS = json.loads((ROOT / "runs/periodic-boundary-2026-09-08/output-loops.json")
                   .read_text(encoding="utf-8"))


def naive_rows(width, last_time):
    state = {0: 1}
    result = []
    radius = width // 2
    for t in range(last_time + 1):
        result.append(sum(state.get(i, 0) << (i + radius)
                          for i in range(-radius, radius + 1)))
        state = {i: state.get(i - 1, 0) ^
                 (state.get(i, 0) | state.get(i + 1, 0))
                 for i in range(-t - 1, t + 2)}
    return result


class SeedStripCompatibilityTests(unittest.TestCase):
    def test_packed_seed_rows_match_cell_reference_past_bit_64(self):
        got = probe.seed_strip_rows([3, 15], 90)
        self.assertEqual(got[3], naive_rows(3, 90))
        self.assertEqual(got[15], naive_rows(15, 90))

    def test_every_record_starts_from_an_exact_actual_row(self):
        artifact = probe.analyze(LOOPS, 128)
        actual = probe.seed_strip_rows(artifact["widths"], 160)
        for record, certificate in zip(artifact["records"], LOOPS):
            for hit in record["alignments"]:
                cycle = certificate["closed_walks"][hit["choice"]][:-1]
                self.assertEqual(actual[record["width"]][hit["onset"]],
                                 cycle[hit["loop_offset"]][1])

    def test_full_artifact_roundtrip_and_tamper_rejection(self):
        artifact = probe.analyze(LOOPS, 64)
        self.assertTrue(probe.verify(artifact, LOOPS)["verified"])
        changed = copy.deepcopy(artifact)
        changed["records"][0]["exact_initial_alignments"] += 1
        with self.assertRaises(ValueError):
            probe.verify(changed, LOOPS)

    def test_invalid_inputs_rejected(self):
        for widths in ([], [2], [4], [True]):
            with self.assertRaises(ValueError):
                probe.seed_strip_rows(widths, 5)
        with self.assertRaises(ValueError):
            probe.analyze([], 5)


if __name__ == "__main__":
    unittest.main()

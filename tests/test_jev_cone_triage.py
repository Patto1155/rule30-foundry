import copy
import unittest

from tools.jev_cone_triage import build_plan, check_plan


class ConeTriageTests(unittest.TestCase):
    def test_freeze_is_deterministic_and_unknowns_are_explicit(self):
        first = build_plan()
        self.assertEqual(first, build_plan())
        self.assertEqual(len(first['measurements']), 75)
        self.assertTrue(all(all(v is None for v in m.values())
                            for m in first['measurements'].values()))
        check_plan(first)

    def test_posthoc_order_change_is_rejected(self):
        plan = copy.deepcopy(build_plan())
        plan['frozen_design']['baseline_order'].reverse()
        with self.assertRaisesRegex(ValueError, 'frozen'):
            check_plan(plan)

    def test_measurements_need_provenance(self):
        plan = build_plan()
        first = next(iter(plan['measurements'].values()))
        first['state_count'] = 1
        with self.assertRaisesRegex(ValueError, 'evidence source'):
            check_plan(plan)
        first['evidence_source'] = 'example-independent-witness.json'
        check_plan(plan)

import unittest

from experiments.period7_ether_interface import (
    ETHER,
    arrival_half,
    certificate,
    direct_cross_check,
    interface_initial,
    interface_step,
)
from experiments.alternating_left_transfer import half_step


class Period7EtherInterfaceTests(unittest.TestCase):
    def test_ether_is_boundary_compatible_period_four(self):
        self.assertEqual(["".join(map(str, phase)) for phase in ETHER], [
            "1000000", "1100001", "0110010", "1011111"
        ])
        self.assertEqual([ETHER[t][-1] for t in range(4)], [0, 1, 0, 1])


    def test_interface_matches_direct_truncations_at_arrival(self):
        width = 18
        delta = interface_initial(width)
        for time in range(1, 7 * 32 + 1):
            delta = interface_step(delta, time - 1, width)
            if time % 7:
                continue
            r = time // 7
            direct = ((1 << (7 * r)) - 1) // 127
            for t in range(time):
                direct = half_step(direct, t & 1)
            self.assertEqual(
                (arrival_half(delta, time, width) ^ direct)
                & ((1 << width) - 1), 0)


    def test_complete_cycle_certificate_and_direct_values(self):
        report = certificate()
        self.assertEqual(report["pure_sample_cycle_length"], 512)
        self.assertEqual(report["failure_delay_counts"], {
            "0": 256, "1": 128, "2": 64, "4": 32,
            "5": 24, "6": 4, "7": 2, "8": 2,
        })
        self.assertEqual(report["small_even_exact_first_failures"], {
            "2": 8, "4": 16, "6": 21, "8": 28,
            "10": 39, "12": 43, "14": 49,
        })
        self.assertEqual(direct_cross_check(14), [
            3, 8, 10, 16, 17, 21, 24, 28, 31, 39, 38, 43, 45, 49,
        ])

    def test_one_added_bit_refutes_uniform_delay_eight_extension(self):
        r = 1181
        arrival_macrotime = 8254 // 2
        state = ((1 << (7 * r)) - 1) // 127
        state |= 1 << 8254
        arrival_low_pairs = []
        for macrotime in range(arrival_macrotime + 8 + 1):
            self.assertIn(state & 3, (1, 2))
            if macrotime == arrival_macrotime:
                self.assertEqual(state & ((1 << 18) - 1), 120433)
            if macrotime >= arrival_macrotime:
                arrival_low_pairs.append(state & 3)
            state = half_step(half_step(state, 0), 1)
        self.assertEqual(arrival_low_pairs, [1, 2, 2, 1, 1, 2, 1, 2, 1])


if __name__ == "__main__":
    unittest.main()

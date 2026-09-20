#!/usr/bin/env python3
"""Independent exact checks for the two interface-obstruction notes.

This module uses the elementary Rule 30 truth table directly.  In particular,
it imports neither an experiment/certificate producer nor a packed Rule 30
kernel.  For the generic triangular-map induction, software checks its Boolean
base and cancellation step; the quantified induction remains a mathematical
argument in the accompanying note rather than a machine-checked theorem.
"""

from __future__ import annotations

import argparse
import json
import math


def cell(rule: int, left: int, centre: int, right: int) -> int:
    """Evaluate an elementary cellular automaton truth table."""
    return (rule >> (4 * left + 2 * centre + right)) & 1


def ether_orbit() -> tuple[tuple[int, ...], ...]:
    """Return the four temporal phases of the period-seven Rule 30 ether."""
    row = tuple(int(x) for x in "1000000")
    phases = []
    for _ in range(4):
        phases.append(row)
        row = tuple(cell(30, row[(j + 1) % 7], row[j], row[(j - 1) % 7])
                    for j in range(7))
    if row != phases[0]:
        raise AssertionError("period-seven ether does not close in four steps")
    return tuple(phases)


ETHER = ether_orbit()


def bit(word: int, index: int) -> int:
    return (word >> index) & 1 if index >= 0 else 0


def interface_step(delta: int, time: int, cut: int, width: int) -> int:
    """Advance an exact defect prefix in coordinates ``cut-time+d``.

    Including ``cut`` is essential for the formal cut -2.  Dropping it happens
    to work for cuts divisible by seven but gives the wrong inverse states here.
    """
    phase = ETHER[time % 4]
    following = 0
    for d in range(width):
        ether = (
            phase[(cut - time + d) % 7],
            phase[(cut - time - 1 + d) % 7],
            phase[(cut - time - 2 + d) % 7],
        )
        actual = (
            ether[0] ^ bit(delta, d),
            ether[1] ^ bit(delta, d - 1),
            ether[2] ^ bit(delta, d - 2),
        )
        following |= (cell(30, *actual) ^ cell(30, *ether)) << d
    return following


def advance_interface(delta: int, start: int, count: int,
                      cut: int, width: int) -> int:
    for time in range(start, start + count):
        delta = interface_step(delta, time, cut, width)
    return delta


def inverse_triangular_apply(forward, target: int, width: int) -> int:
    """Invert a unitriangular prefix map successively from low bit to high."""
    source = 0
    for d in range(width):
        lower_output = bit(forward(source), d)
        source |= (lower_output ^ bit(target, d)) << d
    if forward(source) != target:
        raise AssertionError("purported unitriangular inverse did not invert")
    return source


def verify_defect_algebra() -> None:
    """Exhaust the five-variable Boolean identity used by recurrence (1)."""
    for left in (0, 1):
        for centre in (0, 1):
            for right in (0, 1):
                for dm1 in (0, 1):
                    for dm2 in (0, 1):
                        for d in (0, 1):
                            direct = (cell(30, left ^ d, centre ^ dm1,
                                           right ^ dm2) ^
                                      cell(30, left, centre, right))
                            formula = (d ^ ((1 ^ right) & dm1) ^
                                       ((1 ^ centre) & dm2) ^ (dm1 & dm2))
                            if direct != formula:
                                raise AssertionError("defect ANF identity failed")


def triangular_exponent_bound(width: int) -> int:
    """Check the Boolean base and cancellation step of the stated induction.

    If the first ``width-1`` coordinates have exponent ``N``, the Nth power
    fixes them.  Its last coordinate is therefore ``x XOR sigma(lower)``;
    applying that map twice cancels sigma.  The two-entry truth table below is
    the only Boolean-algebra fact used by the induction.
    """
    if width < 1:
        raise ValueError("width must be positive")
    if any(((x ^ sigma) ^ sigma) != x
           for x in (0, 1) for sigma in (0, 1)):
        raise AssertionError("top-coordinate translation is not an involution")
    exponent = 1  # width one: bit zero is fixed
    for prefix_width in range(2, width + 1):
        previous = exponent
        exponent *= 2
        if exponent != 2 * previous or exponent != 1 << (prefix_width - 1):
            raise AssertionError("unitriangular exponent induction failed")
    return exponent


def small_return_map_order(width: int) -> int:
    """Compute the cut-26 return-map order on a deliberately small prefix."""
    if not 1 <= width <= 12:
        raise ValueError("small exhaustive order check is limited to 12 bits")
    following = [advance_interface(x, 0, 28, 26, width)
                 for x in range(1 << width)]
    seen = set()
    order = 1
    for initial in range(1 << width):
        if initial in seen:
            continue
        state = initial
        cycle_length = 0
        while state not in seen:
            seen.add(state)
            cycle_length += 1
            state = following[state]
        order = math.lcm(order, cycle_length)
    return order


def physical_step(cells: list[int], time: int, rule: int = 30) -> list[int]:
    """One half-line step in a shrinking cone with boundary ``time mod 2``."""
    return [cell(rule, cells[j + 1], cells[j],
                 (time & 1) if j == 0 else cells[j - 1])
            for j in range(len(cells) - 1)]


def full_row_first_centre_failure(radius: int, delete_left_bit_42: bool,
                                  horizon: int) -> int | None:
    """Replay a finite full row in the exact shrinking centre cone."""
    left_support = {-1 - 7 * j for j in range(radius)}
    right_support = {6 + 7 * j for j in range(radius)}
    if delete_left_bit_42:
        left_support.remove(-43)
    cells = [int(x in left_support or x in right_support)
             for x in range(-horizon, horizon + 1)]
    for time in range(horizon + 1):
        if cells[len(cells) // 2] != (time & 1):
            return time
        if time != horizon:
            cells = [cell(30, cells[j - 1], cells[j], cells[j + 1])
                     for j in range(1, len(cells) - 1)]
    return None


def macro_residues(prefix: int, last_macrotime: int,
                   rule: int = 30) -> list[int]:
    """Low-pair residues through ``last_macrotime`` by cell-level replay."""
    cells = [bit(prefix, j) for j in range(2 * last_macrotime + 2)]
    residues = []
    for macrotime in range(last_macrotime + 1):
        residues.append(cells[0] | (cells[1] << 1))
        if macrotime != last_macrotime:
            cells = physical_step(cells, 2 * macrotime, rule)
            cells = physical_step(cells, 2 * macrotime + 1, rule)
    return residues


def verify_formal_survivor(width: int = 24) -> dict:
    """Check phases, inverse time, and the boundary-zero ether merge."""
    if width < 3:
        raise ValueError("width must be at least three")
    cut = -2
    if interface_step(3, -2, cut, width) != 1:
        raise AssertionError("formal time -2 state does not advance 3 -> 1")
    if interface_step(1, -1, cut, width) != 1:
        raise AssertionError("formal time -1 state does not advance 1 -> 1")

    # Check that the same state is A_26 Q^{-1}(1), including composition order.
    q = lambda x: advance_interface(x, 0, 28, 26, width)
    q_inverse_one = inverse_triangular_apply(q, 1, width)
    if advance_interface(q_inverse_one, 0, 26, 26, width) != 3:
        raise AssertionError("A_26 Q^-1(1) is not the claimed arrival state")

    # The imposed boundary agrees with the ether before arrival.  Thus the
    # moving recurrence is valid through the step that reaches cell zero.
    if any(ETHER[t % 4][-1] != (t & 1) for t in range(28)):
        raise AssertionError("alternating boundary is incompatible with ether")

    phase_two = [ETHER[2][j % 7] ^ (1 if j < 2 else 0)
                 for j in range(width + 1)]
    after = [cell(30, phase_two[j + 1], phase_two[j],
                  0 if j == 0 else phase_two[j - 1])
             for j in range(width)]
    phase_three = [ETHER[3][j % 7] for j in range(width)]
    if after != phase_three:
        raise AssertionError("T_0(E_2 XOR 3) does not merge into E_3")
    # Only output indices 0,1,2 can see either changed arrival bit.
    affected_outputs = tuple(range(3))
    return {
        "formal_defects_at_times_minus2_minus1_zero": [3, 1, 1],
        "arrival_defect": 3,
        "arrival_ether_phase": 2,
        "first_boundary_after_arrival": 0,
        "merge_phase": 3,
        "possibly_affected_merge_outputs": list(affected_outputs),
    }


def verify_approximation_samples(max_width: int = 10) -> None:
    """Small direct phase checks; the all-width result uses the induction."""
    for width in range(1, max_width + 1):
        k = max(0, width - 1)
        cut = 28 * (1 << k) - 2
        arrival = advance_interface(1, 0, cut, cut, width)
        if arrival != (3 & ((1 << width) - 1)):
            raise AssertionError("finite 2-adic approximant has wrong arrival")


def verify_fixed_hole() -> dict:
    """Check q(n_6)=21 and the exact prefix locality used for every R>=7."""
    n6 = sum(1 << (7 * j) for j in range(6))
    residues = macro_residues(n6, 21)
    if residues != [1, 2] * 10 + [1, 3]:
        raise AssertionError("n_6 low-pair orbit is incorrect")
    if any(value not in (1, 2) for value in residues[:21]) or residues[21] != 3:
        raise AssertionError("q(n_6) is not 21")

    # A latch at macrotime h uses exactly initial indices 0,...,2h+1.
    dependency_max = 2 * 21 + 1
    for radius in range(7, 13):
        n_r = sum(1 << (7 * j) for j in range(radius))
        with_hole = n_r ^ (1 << 42)
        mask = (1 << (dependency_max + 1)) - 1
        if (with_hole & mask) != n6:
            raise AssertionError("fixed-hole prefix does not equal n_6")
        if macro_residues(with_hole, 21) != residues:
            raise AssertionError("fixed-hole locality replay disagrees")
    return {
        "n6": n6,
        "n6_hex": hex(n6),
        "q_n6": 21,
        "residues_through_q": "".join(map(str, residues)),
        "dependency_indices_through_failure": [0, dependency_max],
        "fixed_deleted_bit": 42,
        "next_possible_outer_one": 49,
        "all_R_at_least_7_q_with_hole": 21,
    }


def verify_small_edit_family() -> dict:
    """Check the bit-18 arrival word, delay ten, and its finite cone transfer."""
    edit = 18
    failure_macrotime = 19
    dependency_max = 2 * failure_macrotime + 1

    # The isolated periodic-tail configuration needs exactly indices 0..39 to
    # reconstruct arrival indices 0..21 after 18 halfsteps.
    cells = [ETHER[0][j % 7] ^ (1 if j == edit else 0)
             for j in range(dependency_max + 1)]
    for time in range(edit):
        cells = physical_step(cells, time)
    if len(cells) != 22:
        raise AssertionError("bit-18 arrival cone did not shrink to 22 bits")
    arrival_bits = "".join(map(str, cells))
    expected_arrival = "1010100001000001011010"
    if arrival_bits != expected_arrival:
        raise AssertionError("bit-18 arrival prefix is incorrect")
    arrival_word = sum(value << j for j, value in enumerate(cells))
    residues = macro_residues(arrival_word, 10)
    expected_residues = [1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 0]
    if residues != expected_residues:
        raise AssertionError("bit-18 post-arrival residues are incorrect")

    # For every r>=6, the first omitted ether one is at 7r>=42, beyond index
    # 39.  A few representatives are replayed; the all-r step is the displayed
    # support inequality and radius-one locality.
    n6 = sum(1 << (7 * j) for j in range(6))
    finite_prefix = n6 | (1 << edit)
    mask = (1 << (dependency_max + 1)) - 1
    for radius in range(6, 12):
        n_r = sum(1 << (7 * j) for j in range(radius)) | (1 << edit)
        if (n_r & mask) != finite_prefix:
            raise AssertionError("bit-18 finite truncation changes its cone")
        direct = macro_residues(n_r, failure_macrotime)
        if (any(value not in (1, 2) for value in direct[:failure_macrotime]) or
                direct[failure_macrotime] not in (0, 3)):
            raise AssertionError("bit-18 finite truncation does not have q=19")
    return {
        "edited_bit": edit,
        "arrival_macrotime": edit // 2,
        "arrival_prefix_width": 22,
        "arrival_bits_low_index_first": arrival_bits,
        "post_arrival_residues_delays_0_through_10": residues,
        "post_arrival_failure_delay": 10,
        "q_for_all_r_at_least_6": 19,
        "initial_dependency_indices_through_failure": [0, dependency_max],
        "first_omitted_ether_one_lower_bound": 42,
    }


def verify_physical_full_row_corollary() -> dict:
    """Check the actual centre-trace gain after restoring left bit 42."""
    # This is the physical-coordinate ether, with ones at x == 6 (mod 7).
    phase = tuple(int(x == 6) for x in range(7))
    centre_trace = []
    for time in range(4):
        centre_trace.append(phase[0])
        phase = tuple(cell(30, phase[(x - 1) % 7], phase[x],
                           phase[(x + 1) % 7]) for x in range(7))
    if centre_trace != [0, 1, 0, 1]:
        raise AssertionError("physical period-seven ether is not alternating")

    representative = {}
    for radius in (7, 8, 9):
        restored_expected = 7 * radius + 1
        restored = full_row_first_centre_failure(
            radius, delete_left_bit_42=False, horizon=restored_expected)
        deleted = full_row_first_centre_failure(
            radius, delete_left_bit_42=True, horizon=43)
        if restored != restored_expected or deleted != 43:
            raise AssertionError("full-row shrinking-cone replay is incorrect")
        representative[str(radius)] = {
            "restored_first_failure": restored,
            "deleted_first_failure": deleted,
        }

    # Half-line bit j is physical x=-1-j, so bit 42 is x=-43.  For R>=7 it
    # is the unique closest discrepancy after deletion.  With it restored,
    # the missing ether sites nearest the centre are -(7R+1) and 7R+6.
    # Radius-one causality excludes an earlier failure.  Iterated left
    # permutivity makes the extreme left-cone derivative one, forcing failure
    # exactly when the left discrepancy reaches the centre.
    if any(cell(30, 0, centre, right) == cell(30, 1, centre, right)
           for centre in (0, 1) for right in (0, 1)):
        raise AssertionError("Rule 30 is not left permutive")
    return {
        "physical_ether_one_residue_mod_7": 6,
        "ether_centre_trace_period": centre_trace,
        "left_half_bit_42_physical_site": -43,
        "restored_nearest_missing_sites": ["-(7R+1)", "7R+6"],
        "restored_exact_first_failure": "7R+1",
        "deleted_exact_first_failure_for_R_at_least_7": 43,
        "restoration_prefix_gain": "7R-42",
        "representative_shrinking_cone_replays": representative,
        "claim_scope": (
            "Exact finite centre-trace prefix only; no eventual-trace claim."
        ),
    }


def verify() -> dict:
    verify_defect_algebra()
    exponent = triangular_exponent_bound(28)
    formal = verify_formal_survivor()
    verify_approximation_samples()
    fixed_hole = verify_fixed_hole()
    small_edit = verify_small_edit_family()
    physical = verify_physical_full_row_corollary()
    return {
        "verified": True,
        "rule30_truth_table": True,
        "q28_prefix_exponent_divides": exponent,
        "q_return_map_actual_order_width_4": small_return_map_order(4),
        "prefix_L_exponent_formula": "2^(L-1)",
        "induction_check_scope": (
            "Executable checks cover the Boolean base and cancellation step; "
            "the all-L induction is a mathematical argument, not a formal proof."
        ),
        "finite_approximation_condition": "k >= L-1",
        "horizon_prefix_width": "L=2H+2",
        "finite_transfer_condition": "7r > p_k+2H+1",
        "formal_survivor": formal,
        "fixed_hole": fixed_hole,
        "small_bit_18_family": small_edit,
        "physical_full_row_corollary": physical,
        "scope": (
            "The first construction disproves a uniform post-arrival bound "
            "for isolated wakes and finite one-bit ether modifications. The "
            "fixed-hole construction disproves a bounded marginal insertion "
            "allowance, while relying on the established odd-R truncation formula."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(verify(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

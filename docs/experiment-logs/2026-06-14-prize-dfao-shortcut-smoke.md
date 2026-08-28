# Experiment Log

- Date: 2026-06-14
- Title: Prize DFAO Shortcut Smoke Test
- Claim Level: Certificate for the tested finite prefixes and state bounds
- Goal: Test a named shortcut class for Wolfram prize problem 2: binary DFAO
  programs computing `n -> center_bit(n)` over the first 128 single-black-cell
  Rule 30 center-column bits.
- Hypothesis: If the center column has a very small automatic shortcut, then a
  1-4 state binary DFAO should fit the verified 128-bit prefix in either
  most-significant-digit or least-significant-digit input order.
- Prediction: A positive-control automatic sequence should be found by the same
  search. If Rule 30 has no such tiny shortcut, exhaustive search returns no fit
  through 4 states.
- Setup:
  - Tool: `prize_lab.py dfao-search`
  - Prefix length: 128 bits, including time/index 0
  - Rule 30 source artifact: `data/prize/center_prefix_n128_verified.json`
  - Positive control: Thue-Morse, which has a 2-state binary DFAO
  - Baseline/null: seeded random 128-bit prefix, seed 30
  - Directions: `msd` and `lsd` for the Rule 30 center prefix and random baseline
- Verification:
  - `python_int`, packed CPU, and naive-array Rule 30 center-column engines agree
    over the 128-bit center prefix.
  - Positive-control candidate was checked with `prize_lab.py check-dfao`.
- Commands:
  - `python prize_lab.py center --steps 127 --no-include-bits --out data/prize/center_prefix_n128_verified.json`
  - `python prize_lab.py dfao-search --sequence thue-morse --bits 128 --max-states 2 --direction msd --out data/prize/dfao_control_thue_morse_b2_msd_n128_s2.json`
  - `python prize_lab.py check-dfao --candidate data/prize/dfao_control_thue_morse_b2_msd_n128_s2.json --sequence thue-morse --steps 127 --out data/prize/dfao_control_thue_morse_b2_msd_n128_s2_check.json`
  - `python prize_lab.py dfao-search --sequence center --bits 128 --max-states 4 --direction msd --out data/prize/dfao_center_b2_msd_n128_s4.json`
  - `python prize_lab.py dfao-search --sequence center --bits 128 --max-states 4 --direction lsd --out data/prize/dfao_center_b2_lsd_n128_s4.json`
  - `python prize_lab.py dfao-search --sequence random --bits 128 --max-states 4 --direction msd --seed 30 --out data/prize/dfao_random_seed30_b2_msd_n128_s4.json`
  - `python prize_lab.py dfao-search --sequence random --bits 128 --max-states 4 --direction lsd --seed 30 --out data/prize/dfao_random_seed30_b2_lsd_n128_s4.json`
- Result:
  - Positive control passed: Thue-Morse fit a 2-state DFAO, transitions
    `[[0, 1], [1, 0]]`, outputs `[0, 1]`, and the verifier accepted all 128 bits.
  - Rule 30 center prefix: no binary DFAO with 1, 2, 3, or 4 states fit the
    128-bit prefix in `msd` order.
  - Rule 30 center prefix: no binary DFAO with 1, 2, 3, or 4 states fit the
    128-bit prefix in `lsd` order.
  - Random seed-30 baseline: same no-fit result through 4 states in both
    directions.
- Interpretation: This is a small finite obstruction, not a proof of
  non-automaticity and not evidence stronger than the random baseline. Its value
  is methodological: the repo now has a positive-control-checked, reproducible
  way to turn tiny shortcut classes into checkable artifacts.
- Next Step: Increase the prefix/state frontier using SAT or a smarter canonical
  DFAO search, and keep the random baseline at the same prefix/state budget.

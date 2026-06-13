# Coarse-Grain Search — Same-Local-Statistics Null (Experiment T2)

- **Date:** 2026-06-13
- **Script:** `experiments/coarse_grain_rule_null.py` (reuses `enumerate_b2` from
  `coarse_grain_search.py`; fields from `experiments/eca_sim.py`)
- **Data:** `data/coarse_grain_rule_null.json`

## Goal

Experiment T reported that Rule 30's best b=2 coarse field "closes" with excess
≈ 0.22 over an i.i.d. fair-coin null. An i.i.d. coin is a weak null — *any*
locally-correlated field beats it, because adjacent coarse cells share fine cells
and inherit local rule structure. Decide whether the 0.22 excess is **Rule-30-
specific approximate reducibility** or **generic local leakage** common to all
chaotic elementary CA.

## Setup

Run the identical pipeline (sheared b=2 projection enumeration, excess closure of
the optimal local predictor, neighbourhood r=1, H_min=0.85, shears {0, 0.25, 1})
on several rules, comparing best excess:

- **rule 30** — subject (chaotic, class 3; the canonical irreducible rule).
- **rule 45** — same-statistics null: also chaotic/class-3, ~balanced density.
- **rule 90** — positive control: additive/linear (XOR of neighbours), genuinely
  reducible; a meaningful metric must rate it *above* 30.
- **rule 110** — class-4, Turing-complete.
- **i.i.d.** — the original weak null, for reference.

Fields are random-IC bulk windows with the boundary trimmed.

## Verification (per the implementation guardrails)

- `eca_sim.py` (the arbitrary-rule simulator used to generate every field) is
  verified: packed-CPU == naive cell-by-cell for rules 30/45/90/110/184/60/150
  over 120 steps **across 64-bit word boundaries**; its rule-30 output is
  identical to the trusted `rule30_open_utils.simulate_spacetime`; and GPU == CPU
  for every rule (multi-chunk).
- Sanity check on the metric itself: the additive rule 90 must close at ~1.0 (it
  has an exact coarse-graining); the i.i.d. field must score ~0. Both hold (below)
  — so the metric is calibrated at both ends before we read the Rule 30 value.

## Result (full scale, 1200×1200; matches test scale 400×400)

| rule | best excess | closure | shear | role |
|---|---|---|---|---|
| 30  | +0.2154 | 0.716 | 0.0 | subject (chaotic) |
| 45  | +0.2176 | 0.720 | 0.0 | same-statistics null (chaotic) |
| 90  | **+0.5000** | **1.0000** | 0.0 | positive control (linear → reducible) |
| 110 | +0.3806 | 0.885 | 0.0 | class-4 |
| iid | +0.0174 | 0.570 | 1.0 | weak null |

- **30 − 45 gap = −0.002** (indistinguishable — Rule 30 is, if anything, very
  slightly *below* the chaotic null; the test-scale gap was +0.006, so it
  brackets zero).
- **90 − 30 gap = +0.285** (rule 90 closes perfectly, every run).
- Rule 110 (class-4) sits higher (+0.38) than the class-3 rules — a separate
  observation worth a later look, not the subject here.
- Best shear is axis-aligned (σ=0) for every rule; cone-aligned shear did not
  help — consistent with the prior finding that sheared supercells fail at b=2.

## Interpretation

Two facts together settle the question:

1. **The metric works.** Rule 90, which is exactly reducible by linearity, closes
   at 1.0000 while i.i.d. scores ~0. So the excess-closure probe genuinely
   separates reducible from structureless fields.
2. **Rule 30 is not special.** Rule 30 ≈ rule 45 (both chaotic) at ≈0.22. The
   excess is therefore **generic local leakage shared by chaotic elementary
   rules**, not evidence of Rule-30-specific approximate reducibility.

This is a higher-level empirical finding (a property of the coarse-grain *metric*
applied across rules), not a direct restatement of the local rule. It is
consistent with Israeli–Goldenfeld: Rule 30 has no exact coarse-graining, and at
b=2 it shows no *approximate* one beyond what any chaotic rule trivially exhibits.

**Verdict: the b=2 coarse-grain route to a Rule 30 shortcut is closed.**

## Next step

- The open question moves to **b ≥ 3**, where enumeration (2^512 projections) is
  impossible and a learned straight-through projection search is required.
- Prerequisite: the b=2 enumeration is ~15 s/field (65,534-projection Python
  loop, M-capped). Before b=3, **vectorize via a joint block-code transition
  histogram** (precompute the 16⁴ tuple histogram once, evaluate each π by
  aggregating it through the projection) so the search is GPU-tractable.
- Re-use rule 90 as the standing positive control at every new b.

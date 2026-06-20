# b=3 Coarse-Grain Reducibility Verdict (Experiment U)

- **Date:** 2026-06-14
- **Script:** `experiments/coarse_grain_bk_verdict.py` (search engine in
  `experiments/coarse_grain_bk.py`; drivable via `ca_lab.py search`)
- **Data:** `data/coarse_grain_b3_verdict.json`

## Goal

Extend the b=2 verdict (Experiment T2 — Rule 30's coarse "closure" is generic
chaotic leakage, not reducibility) to **b=3**. At b=3 a projection maps a 3×3
block (512 patterns) to one bit, so there are **2⁵¹² projections** — enumeration
is impossible. Closure is therefore *maximized by search*, which forces two extra
controls (below).

## Setup

- Fields: random-IC bulk windows, 1500×1500, boundary trimmed.
- Coarse closure: optimal local predictor of the next coarse cell from its r=1
  coarse neighbourhood (`coarse_grain_bk.closure_batch`, GPU-resident, general-b).
- Search: GPU (μ+λ) evolutionary hill-climb (`search_projection`), budget 300,000
  scored projections, search on 80k transitions, **best projection re-scored on
  the full uncapped transition set**. Shears {0, 0.25, 1}.

## Controls (mandatory for a search, not an enumeration)

- **Validity gate — shift rule 170.** A pure shift has coarse field
  `coarse[t+1,x]=coarse[t,x+1]` for *any* projection, so closure must reach ~1.0
  at any b/r. If the search can't recover it, it is too weak to trust its
  negatives. *(Linear rules 90/150 are NOT valid b=3 controls — not exactly
  coarse-grainable at r=1; their perfect b=2 closure was special to 2× sublattice
  alignment. Established while building this — see the Phase 1 commit.)*
- **Searched i.i.d. null.** A search overfits finite samples, so the i.i.d. field
  is searched at the same budget/M; its closure is the floor to clear.
- **Same-statistics null — rule 45** (chaotic, class 3), searched at equal budget.

## Verification

- `closure_batch` (the b=3 evaluator) matches the trusted b=2 enumeration
  bit-for-bit (Phase 0 gate).
- The b=3 pipeline is validated end-to-end by the shift control reaching exactly
  1.0 (below).

## Result (best closure over shears; full-M re-score)

| entity | closure | excess | role |
|---|---|---|---|
| shift rule 170 | **1.0000** | +0.499 | validity gate (must ~1.0) — PASS |
| rule 110 | 0.9519 | +0.452 | class-4 (notably coarse-grainable) |
| **rule 45** | 0.7865 | +0.286 | chaotic same-statistics null |
| **rule 30** | 0.7395 | +0.239 | subject |
| i.i.d. | 0.5258 | +0.000 | searched overfitting floor |

- gate = 1.0000 (PASS); i.i.d. floor = 0.526.
- **gap (30 − 45) = −0.047**: Rule 30 closes *below* the chaotic null.

## Interpretation

The gate passes (search recovers the exactly-coarse-grainable shift rule), so the
negative is trustworthy. Rule 30 (0.740) is **not above** rule 45 (0.786) — it is,
if anything, the *more* irreducible of the two, and both sit far below the
class-4 rule 110 (0.952). Only a *positive* 30 − 45 gap would indicate
30-specific reducibility; a negative gap is the opposite.

Across scales the sign is consistent: gap = −0.002 (b=2), −0.018 (b=3 test),
−0.047 (b=3, full). Rule 30 never rises above the chaotic null.

**Verdict: the b=3 coarse-grain route is CLOSED.** Rule 30 shows no
block-coarse-graining shortcut at b∈{2,3} beyond generic chaotic local structure,
and is among the least coarse-grainable rules tested.

### A note on method (discipline catching itself)

The *numbers* were clean, but the auto-verdict first mislabeled the −0.047 gap as
"candidate reducibility" because it tested `abs(gap)` rather than the signed gap.
Fixed: only a gap **> +0.02** (Rule 30 above the null) counts as a candidate.
Lesson logged in `docs/WORKFLOW.md`.

## Next step

- A heuristic-search negative is suggestive, not proof (can't certify "no π exists"
  over 2⁵¹²). Strengthen by: larger r (coarse neighbourhood), multiple field seeds
  for a confidence band, and a gradient/STE cross-check of the search optimum.
- b=4 is mechanically supported (`closure_batch` is general-b) but the search space
  grows as 2^(b²); budget and population must scale accordingly.
- Class-4 rule 110's high closure (0.95) is a separate, interesting thread.

# Experiment Log — Column Mutual Information and Transfer Entropy

- Date: 2026-04-01
- Title: Column MI and Transfer Entropy — first 2D analysis
- Goal: Detect spatial structure and directional information flow invisible to 1D analysis
- Setup: Fresh 500,000-step simulation, strip center+-512 (1025 cols), burn-in=1012, GPU=True
- Method:
  - MI(center, right+d) and MI(center, left-d) for d in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
  - TE(right+d->center) and TE(left-d->center): H(Y_{t+1}|Y_t) - H(Y_{t+1}|Y_t,X_t)
  - Key test: TE asymmetry (TE_right - TE_left) — Rule 30 is NOT left-right symmetric
- Results:
  - n_samples (post burn-in): 498,988
  - Verification: PASS
  - Max |TE asymmetry|: 0.50072854 bits
  - Distances with nonzero asymmetry: 4/10
  - MI at d=1: right=0.000001, left=0.000000
  - TE at d=1: right=0.000003, left=0.500731, asym=-0.500729
- Interpretation:
  TE asymmetry detected at multiple distances — direct empirical measurement of Rule 30's broken left-right symmetry in information propagation.
  MI decays with distance — consistent with mixing (exponential decay expected for ergodic system).
- Elapsed: 56s

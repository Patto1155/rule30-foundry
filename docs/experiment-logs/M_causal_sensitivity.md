# Experiment Log

- Date: 2026-04-01
- Title: Causal Sensitivity Mapping
- Goal: Map how single-bit perturbations propagate to the center column — tests whether Rule 30's sensitivity is uniform, asymmetric, or has blind spots
- Setup: N_STEPS=10000, MAX_DIST=10000, N_CELLS=20001, BATCH=1000, GPU=True
- Method: For each flip distance D (left and right of center), flip that initial bit and measure the first step at which the center column diverges from the unflipped run
- Result:
  - Never-diverged (left):  34/10001
  - Never-diverged (right): 6179/10001
  - Flips within light cone (T<=D), left:  2654/10001
  - Flips within light cone (T<=D), right: 67/10001
  - Mean left-right asymmetry: 2985.08 steps
- Interpretation: Some flips did NOT affect center in 10000 steps — potential insensitive directions.
  Asymmetric propagation detected — Rule 30 favors one direction.
- Verification: Plot saved to docs/plots/. Three panels: (1) first-divergence scatter vs light cone, (2) causal cone heatmap, (3) left vs right symmetry.
- Elapsed: 38s

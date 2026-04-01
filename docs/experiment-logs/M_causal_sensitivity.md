# Experiment Log

- Date: 2026-04-01
- Title: Causal Sensitivity Mapping
- Goal: Map how single-bit perturbations propagate to the center column — tests whether Rule 30's sensitivity is uniform, asymmetric, or has blind spots
- Setup: N_STEPS=10000, MAX_DIST=10000, N_CELLS=20001, BATCH=1000, GPU=True
- Method: For each flip distance D (left and right of center), flip that initial bit and measure the first step at which the center column diverges from the unflipped run
- Result:
  - Never-diverged (left):  0/10001
  - Never-diverged (right): 6212/10001
  - Boundary-speed arrivals T=D (left):  10001/10001
  - Boundary-speed arrivals T=D (right): 2/10001
  - Causality violations T<D (left/right): 0/0
  - Mean left-right asymmetry on common hits: 2945.23 steps
- Interpretation: No causality violations detected after packed-bit fix.
  Strong left-right asymmetry remains after the implementation fix.
- Verification: Plot saved to docs/plots/. Three panels: (1) first-divergence scatter vs light cone, (2) causal cone heatmap, (3) left vs right symmetry.
- Elapsed: 48s

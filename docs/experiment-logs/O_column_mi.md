# Experiment Log — Column Mutual Information and Transfer Entropy

- Date: 2026-04-01
- Title: Column MI and Transfer Entropy — first 2D analysis
- Goal: Detect spatial structure and directional information flow invisible to 1D analysis
- Setup: Fresh 500,000-step simulation, strip center+-512 (1025 cols), burn-in=1012, GPU=True
- Method:
  - MI(center, right+d) and MI(center, left-d) for d in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
  - TE(right+d->center) and TE(left-d->center): H(Y_{t+1}|Y_t) - H(Y_{t+1}|Y_t,X_t)
  - Surrogate control: time-shift the source column by 9973 samples to estimate a same-marginal noise floor
- Results:
  - n_samples (post burn-in): 498,988
  - Verification: PASS
  - Max |TE asymmetry|: 0.50072854 bits
  - Distances with nonzero asymmetry: 4/10
  - MI at d=1: right=0.000001 (surrogate 0.000002), left=0.000000 (surrogate 0.000001)
  - TE at d=1: right=0.000003 (surrogate 0.000004), left=0.500731 (surrogate 0.000002), asym=-0.500729
- Interpretation:
  The dominant d=1 TE signal survives the surrogate control and is therefore not just a finite-sample artifact.
  Beyond d=1, MI and TE mostly sit near the surrogate baseline, so this experiment does not show strong long-range cross-column dependence at the tested distances.
  A large one-step TE at d=1 is still partly a local-rule fact, not automatically a deep global-structure result.
- Elapsed: 65s

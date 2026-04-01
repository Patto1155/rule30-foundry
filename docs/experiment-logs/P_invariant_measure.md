# Experiment Log - Invariant Measure Analysis

- Date: 2026-04-01
- Goal: Probe whether Rule 30 rows evolved from a Bernoulli(0.5) initial condition remain close to Bernoulli at finite block orders.
- Setup: periodic width=65536, burn_in=512, sampled_rows=256, stride=8, GPU=True
- Method:
  - Simulate Rule 30 on a finite torus from a random Bernoulli(0.5) initial row
  - Verify the packed periodic kernel against a naive periodic implementation
  - After burn-in, sample rows every 8 steps
  - Estimate spatial k-block entropy and KL divergence to the uniform block distribution for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
- Result:
  - Sample density: 0.500028
  - Tail conditional entropy h_k (MM-corrected) at k=18: 0.999998
  - Max KL-to-uniform over tested k: 0.000004
  - Compression proxy: rule30=1.000344, random=1.000344
- Interpretation:
  - Finite-order block statistics are close to Bernoulli(0.5) through the tested orders.
  - This is a random-initial-condition finite-torus approximation to the row process, not a proof of the infinite-volume invariant measure.
- Next Step: If finite-order deviations persist, increase width/sample budget or compare multiple independent random seeds before making any stronger invariant-measure claim.

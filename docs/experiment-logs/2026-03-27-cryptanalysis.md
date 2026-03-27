# Experiment Log

- Date: 2026-03-27
- Title: Cryptanalysis framing — Rule 30 as stream cipher
- Goal: Test whether Rule 30 center column is distinguishable from a true RNG using standard cryptanalytic tests
- Setup: 10M center column bits from Rule 30, GTX 1060 GPU, Python/numpy/scipy
- Method: NIST monobit test, runs test, longest run test, serial correlation (lags 1-1000), distinguishing attack at window sizes 1K-1M
- Result:
  - Monobit: p=0.160304, pass=True
  - Runs: p=0.601833, pass=True
  - Longest run: mean=12.69, expected=13.3
  - Serial correlation: max |r| = 0.001150 at lag 771 (random expectation ~0.000316)
  - Distinguishing: 0/4 window sizes flagged as distinguishable
- Interpretation: Rule 30 passes the tested randomness checks at this scale. No sampled window size was distinguishable from a true RNG, and serial correlation stays near the expected random noise floor.
- Next Step: GF(2) representation search to find transforms that expose hidden structure

# Experiment Log

- Date: 2026-03-27
- Title: Period Search in Rule 30 Center Column (periods 1..1,000,000)
- Goal: Search for any repeating period in the center column — a direct attack on Wolfram Problem 1
- Setup: 10M-bit center column; sampled match test (10,000 positions per period); exhaustive verification for promising candidates
- Method: For each candidate period p, estimate P(bit[i] == bit[i+p]) via sampling; flag periods where match rate > 0.55; verify flagged periods exhaustively
- Result: Best period = 42795 with match rate 0.52330000 (z=4.66). 0 periods exceeded threshold for exhaustive checking.
- Interpretation: No significant period found. Center column appears aperiodic up to period 1M. Bonferroni-corrected significance threshold for 1M simultaneous tests at p<0.01 is z>5.61 (match rate >0.528); best observed z=4.66 falls below this, so no candidate period survives multiple-testing correction.
- Next Step: Extend search to larger periods using FFT-based methods; consider quasi-periodic analysis

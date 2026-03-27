# Experiment Log

- Date: 2026-03-27
- Title: Period Search in Rule 30 Center Column
- Goal: Search for any repeating period p (1..1,000,000) in the 10M-bit center column
- Setup: 10M-bit center column; sampled match test (10,000 positions per period); exhaustive verification for promising candidates
- Method: For each candidate period p, estimate P(bit[i] == bit[i+p]) via sampling; flag periods where match rate > 0.55; verify flagged periods exhaustively
- Result: INCOMPLETE — Phase 1 (sampling scan) completed in 497s. All 1,000,000 periods exceeded the 0.55 match-rate threshold, triggering exhaustive verification on all of them. Phase 2 (exhaustive check) was killed at ~4% progress after exceeding the 15-minute timeout (~3.5 hours estimated total).
- Interpretation: The 0.55 threshold is too low for this data — virtually every period has a match rate slightly above 0.5 (as expected for near-random data). The exhaustive phase is computationally infeasible for 1M candidates. The script needs a higher threshold or a different approach to be practical.
- Next Step: Raise EXHAUSTIVE_THRESHOLD to ~0.6 or higher, or skip exhaustive verification and rely on sampling results with statistical significance testing.

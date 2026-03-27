# Experiment Log

- Date: 2026-03-27
- Title: Block Frequency Analysis of Rule 30 Center Column (k=1..20)
- Goal: Test whether k-bit block frequencies match uniform distribution for each k
- Setup: 10M-bit center column; overlapping k-bit windows; chi-squared goodness-of-fit test
- Method: For each k=1..20, count all 2^k patterns in overlapping windows; compute chi-squared vs uniform expectation; flag p < 0.01
- Result: 20/20 block sizes show significant deviation (p < 0.01). Significant k values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
- Interpretation: Deviations at k=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] — may indicate short-range structure or insufficient data for large k.
- Next Step: Cross-reference any deviations with autocorrelation results; proceed to Markov predictor experiment

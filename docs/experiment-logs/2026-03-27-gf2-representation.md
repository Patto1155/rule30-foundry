# Experiment Log

- Date: 2026-03-27
- Title: GF(2) Representation Search
- Goal: Find linear transforms over GF(2) that reduce entropy of Rule 30 center column (attacks Problem 2)
- Setup: 10M center column bits, window sizes w=8,16,32, random + greedy search over XOR projections
- Method: For each window size, search for k-bit XOR projections (k=1,4) that minimize output entropy. Random search (5000 trials) + greedy local search (10 restarts with bit-flip optimization).
- Result:
  - w=8: best 4-bit entropy 2.0000/4 (apparent reduction 50.00%) — FALSE POSITIVE, see interpretation
  - w=16: best 4-bit entropy 4.0000/4 (reduction 0.00%)
  - w=32: best 4-bit entropy 4.0000/4 (reduction 0.00%)
  - Single-bit XOR projections: max reduction 0.000012 at all window sizes (consistent with fair coin)
- Interpretation: No significant algebraic structure found. The w=8 apparent 50% entropy reduction is a false positive: the random search found a 4x8 transform matrix with rank 2 (two rows linearly dependent on the others), which reduces output entropy mechanically regardless of the input distribution. The decisive results are w=16 and w=32, both showing 0% reduction across 1M+ samples. No GF(2) linear projection finds exploitable structure in Rule 30.
- Next Step: Try non-linear GF(2) projections (degree-2 polynomials over the window bits); ensure future searches check matrix rank before recording entropy to avoid degenerate results

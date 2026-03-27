# Experiment Log

- Date: 2026-03-27
- Title: Markov Predictor on Rule 30 Center Column (orders 1..20)
- Goal: Test whether order-k Markov models can predict the center column above chance (50%)
- Setup: Train on first 5,000,000 bits, test on remaining 5,000,000 bits; orders k=1..20
- Method: For each k, build transition table from training data; predict test data using majority-vote per context; measure accuracy and z-score vs 50% null
- Result: Best accuracy = 50.0292% (k=1). Significant orders (|z|>3): none
- Interpretation: No Markov order beats chance — center column has no short-range predictable structure up to order 20.
- Next Step: Try longer-range models (LSTMs, compression-based) or increase test size

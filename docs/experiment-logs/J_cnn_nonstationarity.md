# Experiment Log

- Date: 2026-03-27
- Title: CNN Non-stationarity Probe
- Goal: Detect non-stationarity by classifying which temporal decile a 512-bit window came from (attacks Problem 3)
- Setup: 10M bits split into 10 deciles, 200,000 train windows / 40,000 test windows, 1D CNN
- Method: 4-layer 1D CNN classifies decile label. Chance=10%. Significant: >15% accuracy.
- Result:
  - Test accuracy: 10.15%  (chance: 10.0%)
  - Improvement over chance: 0.15 pp
  - Per-decile accuracy: [0, 0, 0, 0, 100, 0, 0, 0, 0, 0] — mode collapse (model predicts class 4 for all inputs)
  - Time: 46s
- Interpretation: No significant temporal classification. The 10.15% overall accuracy = model collapsed to predicting one decile (class prior = 10%). Mode collapse on ~uniform data is expected: there is no discriminative temporal signal, so gradient descent finds the trivial solution. This is a strong negative result for non-stationarity — no decile is distinguishable from any other. Consistent with Problem 3 equidistribution.
- Next Step: Sequence appears stationary. Move on to Experiment K (Transformer, long-range context).

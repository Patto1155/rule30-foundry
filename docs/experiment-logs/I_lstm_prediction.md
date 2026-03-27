# Experiment Log

- Date: 2026-03-27
- Title: LSTM Prediction Scaling Law
- Goal: Test whether LSTM with non-linear memory can predict Rule 30 center column better than Markov (Exp H)
- Setup: 5M train / 5M test bits, seq_len=128, hidden_sizes=[32, 64, 128, 256], n_layers=2, epochs=3
- Method: Train LSTM next-bit predictor, measure bits-per-token (BPT) and accuracy on held-out bits. BPT < 1.0 = structure found.
- Result:
  - hidden=32: accuracy=50.0291%, BPT=1.000010, time=93s
  - hidden=64: accuracy=49.9611%, BPT=1.000001, time=210s
  - hidden=128: accuracy=50.0291%, BPT=1.000026, time=1423s
  - hidden=256: accuracy=50.0291%, BPT=1.000002, time=1394s
- Interpretation: Baseline BPT=1.0 (fair coin). Best BPT=1.000001, improvement=-0.000001 bits/token.
  No significant improvement over random — supports computational irreducibility
- Next Step: Run Experiment K (Transformer, larger context) for non-linear + long-range test

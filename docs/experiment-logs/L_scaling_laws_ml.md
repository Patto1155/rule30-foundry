# Experiment Log

- Date: 2026-03-27
- Title: ML Scaling Laws (model size and data size)
- Goal: Test if BPT improves with larger model or more training data (proper scaling law analysis)
- Setup: context_len=256, n_heads=4, n_layers=2, epochs=3
  - Model sweep: d_model=[32, 64, 128, 256] at 5,000,000 training bits
  - Data sweep: d_model=64 at n_data=[500000, 1000000, 2000000, 5000000, 7000000]
- Method: Train Transformer predictor, measure BPT on held-out 3M bits at each scale point
- Result:
  - Model sweep BPT range: 0.000263
  - Data sweep BPT range: 0.000539
  - d_model=32: BPT=1.000065, params=17,185
  - d_model=64: BPT=1.000040, params=67,137
  - d_model=128: BPT=1.000304, params=265,345
  - d_model=256: BPT=1.000115, params=1,054,977
  Data sweep:
  - n_data=500,000: BPT=1.000577
  - n_data=1,000,000: BPT=1.000038
  - n_data=2,000,000: BPT=1.000088
  - n_data=5,000,000: BPT=1.000070
  - n_data=7,000,000: BPT=1.000041
- Interpretation: Flat scaling: BPT range = 0.000263 (model), 0.000539 (data). No improvement with scale — computational irreducibility is robust
- Next Step: If flat, the irreducibility evidence is now complete across architectures, scales, and data sizes

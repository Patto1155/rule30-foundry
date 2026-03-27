# Experiment Log

- Date: 2026-03-27
- Title: Transformer Context Length vs. Complexity
- Goal: Test whether longer context helps a transformer predict Rule 30 center column (attacks Problem 2)
- Setup: 7M train / 3M test bits, d_model=64, n_heads=4, n_layers=2, context_lengths=[64, 128, 256, 512, 1024], epochs=3
- Method: GPT-style next-bit predictor, causal mask, measure BPT at each context length. BPT < 1.0 = structure found.
- Result:
  - context=64: accuracy=49.9705%, BPT=1.000090, time=142s
  - context=128: accuracy=49.9279%, BPT=1.000117, time=187s
  - context=256: accuracy=49.9276%, BPT=1.000129, time=313s
  - context=512: accuracy=49.9280%, BPT=1.000186, time=686s
  - context=1024: accuracy=50.0161%, BPT=1.000027, time=2077s
- Interpretation: Baseline BPT=1.0. Best BPT=1.000027, improvement=-0.000027 bits/token.
  Curve still decreasing at max context: False.
  No improvement over random — computational irreducibility holds for transformer too
- Next Step: If significant, increase d_model or n_layers; if flat, this is strong irreducibility evidence across all tested architectures

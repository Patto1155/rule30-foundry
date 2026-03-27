# ML Experiment Candidates — I, J, K

Generated 2026-03-27. Next experiments after A–H baseline.

---

## Experiment I — LSTM Prediction Scaling Law

**Targets:** Problem 2 (faster algorithm), Problem 3 (bias)

**Hypothesis:** If an LSTM achieves cross-entropy below 1.0 bit/step on held-out bits, it has found structure that Markov order 19 missed — non-linear memory that a look-up table can't represent.

**Method:**
- Train LSTM (1–2 layers, hidden=64/128/256) on first 5M bits as next-bit predictor
- Evaluate on last 5M bits: measure bits-per-token (BPT) and accuracy
- Plot BPT vs hidden size to see scaling curve
- Use teacher-forcing for training, autoregressive for eval (no look-ahead leakage)

**Positive result:** BPT < 0.9999 with z > 5 across 5M test bits, OR curve still decreasing at hidden=256

**GPU:** Low (fits in 6GB, no tensor cores needed). Runtime: 30–90 min.

**Framework:** PyTorch (CUDA SM 6.1 compatible). FP32 recommended (Pascal has no hardware FP16).

---

## Experiment J — Multi-Scale CNN Non-stationarity Probe

**Targets:** Problem 3 (equidistribution), Problem 1 (periodicity as special case)

**Hypothesis:** If the sequence is non-stationary (bias drifts over time), a CNN trained to classify temporal position of windows will perform above chance. Relevant to whether long-run equidistribution holds.

**Method:**
- Frame as 10-class classification: given 512-bit window, predict which decile (of 10M bits) it came from
- 1D CNN: 3–4 conv layers, kernel sizes 3/7/15, ~500K params
- Train on 2M sampled windows, eval on 400K held-out
- Baseline accuracy: 10% (random). Significant result: >15%

**Positive result:** Above-chance temporal position classification confirms non-stationarity

**GPU:** Minimal. Runtime: <15 min. Can interleave with I and K training runs.

---

## Experiment K — Transformer Context Length vs. Complexity

**Targets:** Problem 2 (faster algorithm / shortcut)

**Hypothesis:** A transformer with context L=1024 goes far beyond what Markov order 19 can represent. If BPT keeps falling as L increases (not plateaued), some long-range structure exists.

**Method:**
- GPT-style decoder-only transformer: d_model=64/128, 2–4 layers, 2–4 heads
- Context lengths L ∈ {64, 128, 256, 512, 1024}
- Train on 7M bits, eval on last 3M bits
- Plot BPT vs log(L) — look for continuing decrease past L=512
- Bonus: visualize attention weights on test examples for interpretability

**Positive result:** BPT at L=1024 measurably < BPT at L=64, curve not plateaued

**GPU:** Medium. O(L²) attention. At L=1024, ~10 min/epoch. Total 2–3 hours for full sweep.
**Note:** Avoid `torch.compile` (requires SM 7.0+). Use eager execution.

---

## Priority Order

1. **I (LSTM)** — lowest cost, direct comparison to Exp H Markov results, 30–90 min
2. **K (Transformer)** — highest ceiling, 2–3 hours, run after LSTM as calibration
3. **J (CNN)** — orthogonal, attacks Problem 3, <15 min, can run anytime

## Implementation Notes (GTX 1060 SM 6.1)

- PyTorch works, CuPy already proven
- No `torch.compile`, no tensor cores
- FP16 may be slower than FP32 on Pascal — test both
- 10M bits = 1.25MB RAM — load once as `torch.Tensor(dtype=torch.uint8)`
- Use `torch.unfold` for GPU-batched sequence construction

# rule30-foundry

GPU-powered empirical research into **Wolfram's Rule 30 Prize Problems** — $30,000 each.

![Rule 30 — 55 steps from a single live cell, animated](rule30_animated.svg)

**The question:** Is this sequence random? Does it repeat? Can it be predicted without simulating it?

Nobody knows. Wolfram is offering **$30,000 per problem** to find out.

---

## The Three Prize Problems

| # | Problem | This Project's Result |
|---|---------|----------------------|
| 1 | Does the center column ever **repeat** (become periodic)? | No period found up to 1,000,000 steps |
| 2 | Is there a **faster algorithm** than running Rule 30 step by step? | No linear or Markov shortcut found |
| 3 | Are 0s and 1s **equally distributed** in the long run? | Bias < 0.05% over 10M bits — consistent with yes |

---

## Experimental Results (10M bits, GTX 1060 GPU)

| Exp | Test | Result | Verdict |
|-----|------|--------|---------|
| A | Bit frequency bias | +0.044% bias, 1.4× noise floor | Fair coin ✓ |
| B | Autocorrelation (lags 1–100K) | max \|r\| = 0.00138 | No linear structure ✓ |
| C | k-bit block frequency (k=1..20) | 0/20 sizes deviate | Uniform distribution ✓ |
| D | Markov predictor (order 1–20) | Best accuracy = 50.03% | Unpredictable ✓ |
| E | Period search (p=1..1,000,000) | Best z = 4.66 < 5.61 (Bonferroni) | No period found ✓ |
| F | Cryptanalysis (NIST suite) | Passes monobit, runs, distinguishing attack | Indistinguishable from RNG ✓ |
| G | GF(2) linear transform search | No significant entropy reduction | No algebraic shortcut ✓ |
| H | Markov scaling laws (order 1–18) | Accuracy flat at ~50% across all orders | Computationally irreducible ✓ |
| I | LSTM predictor (hidden 32–256) | BPT=1.000001 at all sizes — no non-linear memory | No LSTM shortcut ✓ |
| J | CNN non-stationarity probe | 10.15% accuracy (chance=10%) — mode collapse | Stationary sequence ✓ |
| K | Transformer (context 64–1024) | BPT flat at ~1.000 across all context lengths | No long-range structure ✓ |
| L | ML scaling laws (model+data) | BPT range <0.001 across d_model=32–256 and n_data=500K–7M | No scaling improvement ✓ |

**All 12 experiments are consistent with Rule 30 being computationally irreducible.**
No architecture (Markov, LSTM, CNN, Transformer) at any scale — of model size, context length, or training data — finds exploitable structure in 10 million bits.

---

## GPU Performance

**Hardware:** GTX 1060 6GB (SM 6.1) · i5-7600K · 16 GB RAM

```
  Rule 30 Simulation  (CuPy CUDA, bit-packed uint64 tape)
  ╔══════════════════════════════════════════╗
  ║  Tape width:   21,000,000 cells          ║
  ║  Steps:        10,000,000                ║
  ║  Throughput:   27,500 steps/s            ║
  ║  Cell rate:    579 Gcells/s              ║
  ║  Runtime:      ~6 minutes                ║
  ╚══════════════════════════════════════════╝

  LLM Inference  (llama.cpp, Qwen2.5-7B Q4_K_M, ngl=35)
  ╔══════════════════════════════════════════╗
  ║  Prompt processing:   414 tokens/s       ║
  ║  Text generation:      23 tokens/s       ║
  ║  VRAM used:           ~4.5 GB / 6 GB     ║
  ╚══════════════════════════════════════════╝
```

Full benchmark report: [`docs/gpu-benchmark.md`](docs/gpu-benchmark.md)

---

## Repo Layout

```
rule30-foundry/
├── gpu/rule30_sim.py          CuPy CUDA kernel — Rule 30 simulation
├── experiments/               A–H: one script per experiment
├── docs/
│   ├── gpu-benchmark.md       Hardware performance report
│   ├── experiment-logs/       Dated result logs (A–H)
│   ├── problem-statements/    Formal writeups per prize problem
│   └── idea-bank/             Future experiment candidates
└── data/                      Binary + CSV outputs (gitignored, kept local)
```

---

## Running It

```bash
# Requirements: numpy scipy tqdm cupy-cuda12x nvidia-cuda-nvrtc-cu12

# 1. Generate center column data (GPU, ~6 min)
python gpu/rule30_sim.py --cells 21000000 --steps 10000000 \
  --center --center-out data/center_col_10M.bin

# 2. Run any experiment
python experiments/bit_distribution.py
python experiments/period_search.py
python experiments/cryptanalysis.py
```

---

## Research Discipline

Empirical results here are not the end goal. Each experiment is designed to either:
- find a **counterexample** (breaking a conjecture), or
- narrow the search space enough to attempt a **formal proof**.

*"Rule 30 is perhaps the most striking example of how a simple rule can produce behavior
that seems highly complex and random."* — Stephen Wolfram

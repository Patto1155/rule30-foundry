# rule30-foundry

GPU-powered empirical research into **Wolfram's Rule 30 Prize Problems** — $30,000 each.

![Cellular automata showcase](data-viz/ca_showcase.svg)

**The question:** Is this sequence random? Does it repeat? Can it be predicted without simulating it?

Nobody knows. Wolfram is offering **$30,000 in prizes for 3 problems** to find out.

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
| I † | LSTM predictor (hidden 32–256) | BPT=1.000001 at all sizes — no non-linear memory | No LSTM shortcut *within this model class* † |
| J | CNN non-stationarity probe | 10.15% accuracy (chance=10%) | Stationary sequence ✓ (powered null — see below) |
| K † | Transformer (context 64–1024) | BPT flat at ~1.000 across all context lengths | No long-range structure *within this model class* † |
| L † | ML scaling laws (model+data) | BPT range <0.001 across d_model=32–256 and n_data=500K–7M | No scaling improvement † |

> **Integrity note (updated 2026-08-30).** The bitstream is verified twice over.
> The 10M center column regenerates **byte-identically** on the post-fix kernel
> (sha256 `6f8670b4...`), and an independent CPU implementation sharing no code,
> no bit-order convention and no tape geometry with `gpu/` reproduces the same
> file byte-for-byte over all 10,000,000 bits
> ([log](docs/experiment-logs/2026-08-30-golden-reference-10M.md)).
>
> **I–L have been re-run on the correct bit order and are no longer retracted.**
> The four scripts had decoded the LSB-first bitstream with numpy's MSB-first
> default, training on the center column with every 8-bit block reversed. Run as
> a paired comparison — both decodes, identical budget, seed and architecture —
> the largest difference across 24 configurations is **0.0024**. The bug did not
> change the conclusions.
> See [`docs/experiment-logs/2026-08-30-rerun-il-bitorder.md`](docs/experiment-logs/2026-08-30-rerun-il-bitorder.md).

† **Detection-power caveat on I, K and L.** The re-run added positive controls
the originals never had, and one failed. On `s[i] = s[i-13] ⊕ s[i-27]` — an
LFSR *fully determined* by 27 bits, well inside a 64-bit context — every model
in the suite scores exactly as it does on a fair coin, across **six** budgets
spanning 5× the parameters, more data, more epochs and a short context where
both taps are trivially visible. The same suite learns a period-31 stream and a
short-lag XOR (`s[i-3] ⊕ s[i-5]`) to ~100% accuracy immediately, so the blind
spot is specific to long-lag parity.

That matters here: Rule 30 is **left-permutive**, so its update is XOR-like, and
Experiment S measured linear complexity `L(n) = n/2`. The structure class most
plausibly present in the center column is the one these models cannot see when
it *is* present. I, K and L therefore report a real absence of learnable
structure *for these architectures*, and cannot exclude XOR-type structure at
moderate lags.

J carries no such caveat: its probe detects a 0.30→0.70 bias ramp at **57.7%**
against a 10% floor, and returns chance on a stationary stream, so its null is
powered.

**A–H are consistent with Rule 30 being computationally irreducible.** The
neural results (I–L) now run on the center column and point the same way, but
they carry the detection-power caveat above: I, K and L are negatives over
model classes that demonstrably miss long-lag XOR structure, which is the
structure Rule 30 is most likely to have. The stronger evidence for
irreducibility is the model-class curve work — `s*(n)` (Certificate) and `g(n)`
— not the neural nulls. The reversal is a
deterministic, position-local recoding rather than a randomisation, so the
conclusions may well survive a re-run — but that has to be demonstrated, not
assumed.

---

## GPU Performance

**Hardware:** GTX 1060 6GB (SM 6.1) · i5-7600K · 16 GB RAM

```text
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

```text
rule30-foundry/
├── data-viz/                  README SVG animations and visualizations
├── gpu/rule30_sim.py          CuPy CUDA kernel — Rule 30 simulation
├── experiments/               A–L: one script per experiment
├── docs/
│   ├── gpu-benchmark.md       Hardware performance report
│   ├── experiment-logs/       Dated result logs
│   ├── problem-statements/    Formal writeups per prize problem
│   └── idea-bank/             Future experiment candidates
└── data/                      Binary + CSV outputs
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

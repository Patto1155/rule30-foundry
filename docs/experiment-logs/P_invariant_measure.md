# Experiment P — Invariant Measure Analysis

**date:** planned 2026-03-28
**status:** not run

## Goal

Determine what probability measure the Rule 30 spacetime diagram converges to, and whether it is the Bernoulli(0.5) measure (i.e., i.i.d. fair coin) or something more structured.

## Why This Matters

Ergodic theory provides the strongest formal framework for classifying dynamical systems. The key question is:

> Does Rule 30, started from a random initial condition, converge to the *same* measure as a fair coin flip? Or does it converge to a different shift-invariant measure with hidden long-range correlations?

All experiments A–L are consistent with Bernoulli(0.5) marginally. But the Bernoulli measure is not the only shift-invariant measure on {0,1}^ℤ. A process can have:
- Correct marginal statistics (bit balance = 0.5) ✓ — we verified this
- Correct pairwise statistics (autocorrelation ≈ 0) ✓ — we verified this
- Correct k-mer statistics for small k ✓ — we verified up to k=16

...and *still* not be Bernoulli, if there are correlations at scale k > 16.

## Setup

### P1 — Empirical Measure Convergence

```python
# For increasing k, estimate the joint distribution of k-tuples
# Compare to Bernoulli(0.5)^k via KL divergence
for k in range(1, 25):
    observed = empirical_kmer_distribution(center_col_46M, k)
    expected = uniform_distribution(k)  # Bernoulli(0.5)^k
    kl_div   = kl_divergence(observed, expected)
    record(k, kl_div, sample_size=len(center_col_46M) - k)
```

**Key:** plot KL divergence vs k. If KL → 0 as k → ∞, the process is Bernoulli. If KL stabilises above 0, a different measure is needed.

**Caveat:** sparse samples at large k inflate KL estimates. Use Miller-Madow correction or Bayesian estimator.

### P2 — Entropy Rate Estimation

The entropy rate h of a stationary process satisfies:
```
h = lim_{k→∞} H(X_k | X_1, ..., X_{k-1})
```

For Bernoulli(0.5), h = 1.0 bits/symbol. Estimate via:
```python
# Plug-in estimator: h_k = H(k-tuples) - H((k-1)-tuples)
for k in range(1, 20):
    h_k = entropy(kmer_dist[k]) - entropy(kmer_dist[k-1])
    record(k, h_k)
# Converges to true entropy rate as k → ∞
```

### P3 — Lempel-Ziv Complexity

LZ complexity is an asymptotically consistent entropy rate estimator:
```python
lz_complexity = len(lz77_compress(center_col_46M))
lz_per_bit    = lz_complexity / len(center_col_46M)
random_baseline = lz_per_bit_of_random_sequence(same_length)
```

## What to Measure

| Metric | Description | Bernoulli(0.5) prediction |
|--------|-------------|--------------------------|
| `kl_div(k)` | KL from uniform k-mer dist | → 0 as k → ∞ |
| `entropy_rate_k` | Conditional entropy at lag k | → 1.0 bits |
| `lz_complexity` | LZ complexity per bit | ≈ 1.0 |
| `measure_label` | Best-fit measure family | Bernoulli vs Markov vs other |

## Theoretical Connection

If entropy rate h < 1.0 bit/symbol, the sequence is *not* Bernoulli — it has long-range correlations regardless of how random it looks at short scales. This would be a finding with formal theoretical weight.

If h = 1.0, this strongly supports (but does not prove) that the invariant measure is Bernoulli(0.5).

## Script

New — `experiments/invariant_measure.py` (to be written)

## Next Step

→ Theory: read `docs/theory/` (pending) before designing further experiments

# Experiment M — Causal Sensitivity / Lyapunov Exponent

**date:** planned 2026-03-28
**status:** not run

## Goal

Measure the rate at which a single-bit perturbation spreads through the Rule 30 spacetime diagram. This gives an empirical Lyapunov exponent — the discrete-system analogue of chaos theory's sensitivity to initial conditions.

This is the most theoretically grounded experiment remaining. Where Experiments A–L asked "does the output look random?", this asks "how does the system *behave dynamically*?"

## Background

In continuous chaotic systems, the Lyapunov exponent λ measures exponential divergence of nearby trajectories. In a CA, the analogue is: flip one bit at t=0, measure how the Hamming distance between the original and perturbed runs grows over time.

For Rule 30 specifically:
- The causal cone is bounded at 1 cell/step (information cannot travel faster than light)
- A perturbation *could* spread maximally (filling the cone) or sub-maximally
- The *distribution* of divergence times characterises the dynamics more richly than a single number
- If divergence is immediate and maximal → strongly mixing, supports irreducibility
- If divergence is slow or patchy → residual structure, worth investigating

## Setup

```python
# Pseudocode
original  = run_rule30(steps=N, width=W, seed=S)
perturbed = run_rule30(steps=N, width=W, seed=S, flip_bit=center)

hamming_distance = [popcount(original[t] XOR perturbed[t]) for t in range(N)]
divergence_time  = first t where hamming_distance[t] > 0
cone_fill_ratio  = hamming_distance[t] / min(2*t+1, W)  # fraction of causal cone filled
```

**GPU approach:** Run both simulations simultaneously on GPU, XOR row by row, count set bits with `cp.count_nonzero` or CUDA `__popc__`. Repeat for k random flip positions and average.

**Parameters:**
- Steps: 10,000–100,000 (short runs, many repetitions)
- Width: 1,000,001 (odd, large enough that boundary doesn't interfere)
- Flip positions: center + 100 random positions
- Repetitions: 1,000

## What to Measure

| Metric | Description |
|--------|-------------|
| `divergence_time` | Steps until first bit differs |
| `cone_fill_t` | Fraction of causal cone filled at time t |
| `saturation_time` | Steps until cone_fill stabilises |
| `lyapunov_proxy` | log(hamming_distance[t]) / t |

## Interpretation

- If `cone_fill` → 1.0 rapidly → maximal sensitivity, strongly mixing
- If `cone_fill` saturates below 1.0 → bounded influence, possible structure
- Distribution of `divergence_time` across flip positions → homogeneity vs. hot/cold spots

## Script

`experiments/causal_sensitivity.py` (exists, may need updating for 46M-scale data)

## Next Step

→ Experiment N: Mutual Information Between Columns

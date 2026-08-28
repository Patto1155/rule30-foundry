# Experiment Log - Linear Complexity of the Single-Seed Center Column

- Date: 2026-06-13
- Title: S — Linear Complexity (GF(2) recurrence) of the seed center column
- Goal: A decisive Prize-2 probe on the **actual prize object** — the center
  column from the single seed `…0001000…`. Does it satisfy *any* linear (GF(2))
  recurrence, i.e. is there an LFSR shortcut that generates it without running
  the CA?
- Why this is the right next probe: Experiments Q/R (and the repo's M /
  diagonal-TE) measure how perturbations spread over **random/perturbed ICs** —
  properties of the rule's sensitivity. But the three prizes are about the
  **single deterministic seed**, which has no ensemble to perturb. So we switch
  to probes that operate directly on that one orbit. Linear complexity is the
  cleanest first one: a sharp null (random ⇒ L(n)≈n/2) and a self-certifying
  positive (a plateau *is* a shortcut).
- Setup: GPU-generated seed center column (65,537 bits). Berlekamp-Massey over
  GF(2), word-parallel via Python big-ints. Linear-complexity profile L(n)
  evaluated at n = 4k, 8k, 16k, 32k.
- Method: `python experiments/linear_complexity.py` (quick: `--test`).
- Observations:

  | n      | L(n)   | L/n    |
  |--------|--------|--------|
  | 4096   | 2049   | 0.5002 |
  | 8192   | 4096   | 0.5000 |
  | 16384  | 8192   | 0.5000 |
  | 32768  | 16384  | 0.5000 |

  - Longest plateau (consecutive bits with no complexity increase): **16 steps**.
  - L(n) = n/2 at every scale — the maximal possible linear complexity.
- Conclusion:
  - The seed center column has **maximal linear complexity**: no GF(2) linear
    recurrence of any order up to n/2 generates it. The LFSR / linear-shortcut
    route to Prize 2 is **closed** on the real prize object. This is stronger
    than repo Experiment G, which only tested a single global linear transform.
  - Combined with R's permutivity note, the empirical picture is: linear and
    "looks-random" structure is exhausted; any surviving shortcut must be
    **nonlinear / algebraic**.
- Next Step (highest-value, prize-tied, seed-specific):
  - Test the seed center column for **automatic-sequence** structure (is it the
    output of a finite automaton reading base-k digits of n?) — a positive
    result is a genuine sub-linear shortcut. Pair with a sheared/affine
    spacetime reparametrization scan for a lower-entropy column than the center
    (Prize-2 "compressible along some direction"), each measured against an
    explicit i.i.d. null. Bias to constructions whose positive result is
    *exploitable or provable*, not another null "looks random ✓".

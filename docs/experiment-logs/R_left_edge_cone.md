# Experiment Log - Leftward Dependence-Cone Edge

- Date: 2026-06-13
- Title: R — Leftward Dependence-Cone Edge (interventional; corrects Q)
- Goal: Measure the *causal* leftward damage-spreading edge of Rule 30 — the
  fastest a single right-side perturbation can reach the center — and test
  whether it converges to a hard cone speed `λ_L < 1` (a real frozen-cone /
  conditional-independence result) or merely relaxes to the bulk rate. This
  corrects Experiment Q, which reported the *median* slope as "the velocity".
- Theory anchor (asymmetry is not put in by hand):
  - Rule 30 ANF: `new[i] = a[i-1] ⊕ a[i] ⊕ a[i+1] ⊕ (a[i]·a[i+1])`.
  - `a[i-1]` enters as a **pure XOR** ⇒ Rule 30 is **left-permutive** (bijective
    in its leftmost input). Hence:
    - RIGHTWARD damage propagates at exactly speed 1 on every background:
      `v_right = 1` is a **theorem**, only sanity-checked here.
    - The uniform Bernoulli(1/2) measure is **invariant**: a random i.i.d.
      initial row stays i.i.d. fair forever, so the center column is *provably*
      50/50 for random ICs. (Numerically confirmed: interior single-cell
      P(1)=0.49996 over 24M samples; 3-blocks within 0.0024 of 0.125.)
    - ⇒ Prize-3 equidistribution is **trivially true for random ICs**; its whole
      difficulty is concentrated in the single deterministic seed `…0001000…`.
  - `a[i+1]` sits inside the OR/AND term ⇒ leftward propagation is *conditional*
    (gated on `a[i]=0`), so `v_left` is the only empirically interesting speed.
- Setup: GPU (CuPy), verified packed open-boundary kernel; naive cross-check
  passed. Distances `d ∈ {30,60,120,240,480,960}`, `N_IC=1500` random ICs each,
  horizon `6d`. For each IC flip the initial cell at `+d` (right of center) and
  record `first_div(d)` = first step the center column changes. Causality ⇒
  `first_div(d) ≥ d`; speed-1 is the null edge. Report edge (min / p1 / p10) and
  bulk (median); velocity for each = `d / arrival`.
- Method: `python experiments/left_edge_cone.py` (quick: `--test`).
- Observations:

  | d   | censored | v_edge(min) | v_p10 | v_median |
  |-----|----------|-------------|-------|----------|
  | 30  | 0.107    | 0.625       | 0.395 | 0.261    |
  | 120 | 0.009    | 0.414       | 0.314 | 0.251    |
  | 240 | 0.003    | 0.354       | 0.288 | 0.244    |
  | 480 | 0.000    | 0.325       | 0.274 | 0.244    |
  | 960 | 0.000    | 0.298       | 0.264 | 0.244    |

  - `v_median` locks onto **0.244** and is flat from d=240 up — a genuine,
    reproducible bulk leftward damage-spreading velocity (a Lyapunov-type speed;
    consistent with known Rule-30 left-speed estimates ~1/4).
  - `v_edge` and `v_p10` **decrease monotonically toward the median** as d grows
    (edge: 0.625→0.298, decrements shrinking). The "fast edge" at small d is a
    finite-size overshoot; the arrival-time distribution **concentrates** on the
    single speed.
- Conclusion:
  - There is **one** leftward velocity, `λ_L ≈ 0.244`, not a hard edge above the
    bulk. The leftward dependence cone is *soft* (sub-linear fluctuations around
    `t ≈ d/λ_L`), so this does **not** yield a clean `λ_L<1` frozen-cone theorem.
  - Honest ceiling note: velocities like this are properties of the **rule's
    sensitivity over random/perturbed ICs**. All three prizes concern the
    **single deterministic seed**, which has no ensemble to perturb — so the
    damage-velocity thread (this, plus the repo's M / diagonal-TE) characterises
    the rule but cannot, by itself, touch the prize object.
- Next Step:
  - Shift to probes that operate on the **single seed orbit** itself. First such
    probe done: linear-complexity / GF(2) recurrence — see
    [`S_linear_complexity.md`](S_linear_complexity.md).

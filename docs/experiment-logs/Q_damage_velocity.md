# Experiment Log - Directional Damage-Spreading Velocity

> **CORRECTION (2026-06-13, same day):** This log's headline "v_left ≈ 0.245"
> was obtained by fitting a line to the **median** arrival time, which is a
> *bulk* rate, not the causal light-cone *edge*. The two are different physical
> quantities. The corrected, convergence-checked analysis is in
> [`R_left_edge_cone.md`](R_left_edge_cone.md): the bulk speed is a genuine
> constant **λ_L ≈ 0.244**, but the *edge* (fastest arrival) relaxes *down*
> toward it as d grows — there is **no hard frozen cone** with an edge strictly
> above the bulk. Also: `v_right = 1.000` is a **theorem** (left-permutivity),
> not a measurement. Read R, not the numbers below, as the result.

- Date: 2026-06-13
- Title: Q — Directional Damage-Spreading Velocity
- Goal: Replace the censoring-prone "mean asymmetry in steps" (Experiment M) and
  the right-blindspot framing with a clean, single-number dynamical invariant:
  the velocity at which a single-bit perturbation ("damage") spreads toward the
  center column, measured separately for the leftward and rightward directions,
  and tested over **random** initial conditions rather than the single spike.
- Setup: GPU (CuPy), verified packed open-boundary kernel from `rule30_open_utils`.
  `N_STEPS=4000`, `N_IC=400` random initial rows, flip distances `d=1..300`,
  seeds `{1,2}`. For each IC, flip one cell at distance `d` left/right of center
  and record the first step at which the center column diverges, `first_div(d)`.
  Causality guarantees `first_div(d) >= d`. The slope of `median first_div(d)`
  vs `d` is `1/v` for that direction.
- Method: `python experiments/damage_velocity.py` (quick check: `--test`).
- Observations:
  - **Rightward damage saturates the light cone.** A LEFT-side flip reaches the
    center at exactly `T = d` for every distance and every random IC: median
    excess delay `first_div - d = 0` throughout. `v_right = 1.000`.
  - **Leftward damage is strictly sub-light-cone.** A RIGHT-side flip arrives
    late, with excess delay growing linearly in `d`. Robust large-`d` fit gives
    `slope ≈ 4.08`, i.e. **`v_left ≈ 0.245`** (test-mode short-horizon estimate
    `0.25`). Reproducible across seeds.
  - This is **not** a single-spike artifact. Experiment M's left side looked
    like a trivial `T=D` line because of the spike's regular left edge; here the
    same `v_right=1` / `v_left≈0.245` asymmetry appears over random ICs, so it is
    a property of the **rule**, not the initial condition.
  - Consistent with the interventional spike result (`first_div_right[1]=
    first_div_right[2]=censored`, `first_div_left[d]=d`) and with the
    correlational diagonal-TE asymmetry (left diagonal carries information, right
    does not), but expressed as one reproducible velocity constant.
- Conclusion:
  - Rule 30 has an intrinsic, quantifiable causal anisotropy:
    **v_right = 1, v_left ≈ 0.245** (≈ 1/4). Information injected on the right of
    the center is throttled on its way in; information on the left arrives at the
    speed limit. This is the interventional/causal counterpart of the well-known
    "regular left edge, chaotic right edge" appearance.
- Next Step:
  - Pin down the asymptotic `v_left` with longer horizons / larger `d` and a
    confidence interval, and test whether it is exactly `1/4` or an irrational
    damage-velocity constant.
  - Map the full damage-velocity profile of the elementary CA neighbourhood
    (does any other rule share `v_right=1, v_left≈1/4`?), and connect the
    one-sided light-cone saturation to a possible proof that the center column is
    independent of initial cells `+1, +2` for all time.

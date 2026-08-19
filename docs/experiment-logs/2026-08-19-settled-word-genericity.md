# Experiment Log — Is the Settled-Word Stream Generic? (Testing the 2^-16 Assumption)

- Date: 2026-08-19
- Title: Empirical test of the genericity assumption underpinning the period-16 power analysis
- Claim Level: **Robust observation** (genericity) + **Theorem** (invertibility of the state map)
- Prize: Problem 3 context. Structural work on the seed orbit, not a shortcut claim.

## Goal

`2026-08-19-period-doubling-criterion.md` rests on one modelling assumption:
past `d ≈ 403` the settled words `w_d` behave like independent uniform 16-bit
words, making a consecutive-word collision a `~2^-16` per-diagonal event.
Everything downstream inherits it — "the old evidence had no power", "expected
first failure at `d ~ 1.3×10^5`", "reaching `10^6` clean is a `~2^-11`
coincidence". It was **asserted, never measured**. Next Step #2 of that log was
to fix that. This does.

## Setup

Sample: words from the O(1) pattern map over `403 < d < 53205` — the range
between the last early zero word and the first real collision, so **no ambiguous
step is ever taken**. `n = 52802`. Validated bit-exact against simulation on the
11597-word overlap (`d < 12000`): **0 mismatches**.

Control: 200 matched uniform-random 16-bit samples of the same length. Verdicts
use **empirical two-sided p-values from the control distribution itself**
(`(k+1)/(m+1)`), not a normal approximation, and are Bonferroni-corrected over
the 5 statistics examined.

Tool: `experiments/word_genericity.py`. Exits non-zero only if the sample fails
to validate against simulation — the statistical verdict is *reported*, not
gated, so this cannot be turned into a pass/fail claim by accident.

## Result 1 — genericity holds

| statistic | observed | expected | z | empirical p |
|---|---|---|---|---|
| worst bit balance (of 16) | 26189 | 26401 | −1.84 | 0.687 |
| parity odd | 26544 | 26401 | +1.25 | 0.154 |
| **all-pairs collisions** | **21164** | **21270.7** | **−0.73** | **0.483** |
| chi² high byte (255 df) | 281.7 | 255.0 | +1.18 | 0.348 |
| chi² low byte (255 df) | 275.8 | 255.0 | +0.92 | 0.373 |

**Significant after Bonferroni: none. No `|z| > 3`.**

The decisive row is the third. The all-pairs collision statistic tests the
`2^-16` rate directly, with `C(52802, 2) ≈ 1.39×10^9` pairs behind it:

```
observed collision rate  1.518e-05
modelled 2^-16           1.526e-05     ->  agreement to 0.5%, z = -0.73
```

Supporting: distinct words 36299 vs 36256 expected (+43); lag-1..64 collisions
42 vs 51.6 expected (no serial structure).

> **The `2^-16` model is confirmed, not assumed.** The power analysis, the
> "expected first failure at `d ~ 1.3×10^5`", and the downgrade of the old
> period-16 evidence all stand on measured ground.

### A methodological note worth keeping

A first pass used only **8** controls and flagged `parity_z` and `chi2_high_z`
as "outside the control band". Both were artifacts: a band built from the
min/max of 8 draws is exceeded by chance roughly 2/10 of the time *per
statistic*, so 2 flags out of 5 is the expected yield under a true null. Raising
to 200 controls and switching to empirical p-values dissolved both
(`p = 0.154`, `p = 0.348`). **Do not read min/max bands from small control sets
as evidence** — this is the same "quote the scale before quoting the result"
discipline that the counting bound and the power analysis each enforce in their
own domain.

## Result 2 — the state map is invertible (THEOREM)

**Lemma C.** On states with `v ≠ 0`, the map `(u, v) ↦ (v, w)` is a **bijection**:
`u` is recovered from `(v, w)` by

```
u[t] = w[t+1] XOR (v[t] OR w[t])
```

*Proof.* Immediate from the recursion `w[t+1] = u[t] XOR (v[t] OR w[t])`,
solving for `u[t]`. ∎

Verified: 199995 random `(u, v)` pairs, **0 failures**.

Two consequences:

1. **The word stream can never be truly i.i.d.** — it is a deterministic orbit
   on a finite state space, visiting states *without replacement*. For
   `n ≈ 5×10^4` against a `2^32` space this is numerically negligible (which is
   why Result 1 comes out clean), but the genericity claim is properly
   "statistically indistinguishable from uniform at this sample size", not
   "independent".
2. **The functional graph is a disjoint union of simple paths and cycles**
   (in-degree ≤ 1, out-degree ≤ 1). This is a strong structural constraint and
   it changes how the conjecture should be attacked — see below.

## Interpretation

Task 2's purpose was to decide whether task 3 rests on solid ground. It does:
the `2^-16` rate is measured, so the predicted failure point near `d ~ 1.3×10^5`
and the `~2^-11` coincidence figure at `10^6` are both trustworthy.

But Result 2 suggests the brute-force walk may not be the best use of the
compute. Because the map is a bijection:

- the orbit **never revisits a state** between zero words, so it cannot quietly
  cycle;
- the whole question is a **reachability problem on a `2^32`-node graph whose
  every node has in-degree and out-degree ≤ 1**;
- the "bad" set is exactly the `2^15 = 32768` diagonal states `(u, u)` with
  `parity(u)` odd (Lemmas A + B).

A `2^32`-bit visited bitmap is 537 MB — feasible on this machine. Deciding
"does the forward orbit from the seed ever reach a bad state" exactly, rather
than sampling `10^6` diagonals and quoting a probability, looks reachable. That
would settle the conjecture outright instead of accumulating more evidence of
the kind this repo has already learned to distrust.

The catch: the orbit leaves the bijective region every time it produces a zero
word (~once per `2^16` diagonals), and each exit needs simulation to resolve. So
the graph decomposes into `~2^16`-long segments joined by simulation-resolved
branch points, and the exhaustive analysis has to account for those joins rather
than assume a single clean orbit.

## Next Step

1. **Scope the exhaustive route before running the `10^6` walk.** Determine the
   segment structure: how many zero words in `d < 10^6`, and whether the
   branch points can be enumerated rather than simulated one at a time.
2. If the exhaustive route does not close, fall back to the walk to `d ~ 10^6`
   as originally planned — it is now well-founded, just weaker than a decision
   procedure.
3. **Do not** re-run this genericity test at larger `n` expecting more; the
   `2^-16` rate is already pinned to 0.5% and the binding constraint is
   structural, not statistical.

## Commands

```bash
python experiments/word_genericity.py --pretty \
    --out data/wedge/word_genericity.json
```

Runs in ~10 s.

## Artifacts

- `data/wedge/word_genericity.json` — sample provenance and simulation
  validation, invertibility check, all statistics with expectations, z-scores,
  200-control band, empirical p-values, and the Bonferroni verdict.

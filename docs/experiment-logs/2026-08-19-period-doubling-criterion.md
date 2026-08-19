# Experiment Log — The Period-Doubling Criterion, and Why Period-16 Was Under-Tested

- Date: 2026-08-19
- Title: Exact trigger for period doubling on the left diagonals; power analysis of the period-16 conjecture
- Claim Level: **Theorem** (Lemmas A and B) + **Robust observation** (the power analysis)
- Prize: Problem 3 context (see `docs/theory/README.md` §0), but this is structural work on the seed orbit, not a shortcut claim.

## Goal

`2026-08-15-structured-wedge-decomposition.md` left this open as a Proof
candidate:

> **Conjecture (period-16).** Every left diagonal is eventually periodic with
> period **dividing 16**. Tested over `d = 0..2999` at `T = 20000`, histogram
> `{1:10, 2:8, 4:37, 8:356, 16:2589}`, maximum 16, zero exceptions. Independent
> replication saw the same ceiling over `d = 768..29491`.

and recorded the missing step as *"bound how often the lemma's period-doubling
branch fires"*. This experiment does that — and the answer changes how much the
existing evidence is worth.

## Setup

Settled diagonal `d` is a 16-bit word `w_d` phase-locked to `t mod 16`. The
diagonal recursion `D_d(t+1) = D_{d-2}(t) ⊕ (D_{d-1}(t) ∨ D_d(t))` lifts to the
pattern map `(w_{d-2}, w_{d-1}) ↦ w_d` (`pattern_map_step` in
`experiments/diagonal_recursion.py`), well defined whenever `w_{d-1} ≠ 0`. The
propagation lemma's doubling branch fires exactly when `w_{d-1} = 0`.

Tool: `experiments/period_doubling.py` (new). Runs in 3.4 s, exits non-zero if
any gate fails. Reuses `left_diagonals` and `pattern_map_step` unchanged.

## Result 1 — the doubling trigger is a consecutive-word collision (THEOREM)

**Lemma A.** For `v ≠ 0`, `pattern_map_step(u, v) = 0` **iff** `u = v`.

*Proof.* `w ≡ 0` forces `0 = w[t+1] = u[t] ⊕ (v[t] ∨ 0) = u[t] ⊕ v[t]` for every
`t`, i.e. `u = v`. Conversely if `u = v` then `w ≡ 0` satisfies the recursion,
and since `v ≠ 0` the one-period composite is constant, so that solution is the
unique eventual one. ∎

Hence **`w_d = 0` iff `w_{d-2} = w_{d-1}`**: the doubling branch fires exactly
at a *collision between consecutive settled words*.

**Lemma B.** At a collision the composite is `x ↦ x ⊕ c` with
`c = parity(w_{d-2})`. The period doubles iff `c = 1`; if `c = 0` it stays. ∎

> **Consequence.** Period-16 holds through `D` **iff every consecutive-word
> collision below `D` has an even-parity predecessor.**

This replaces the vague "bound how often the branch fires" with two conditions
that are checkable without simulating past the seed.

### Verification

| gate | result |
|---|---|
| Lemma A, **exhaustive** at period 8 (all 65280 pairs with `v≠0`) | **0 violations** |
| Lemma A, sampled at period 16 (200k pairs) | **0 violations** |
| Lemma A against simulation, `d = 0..11999` | **0 violations** |
| Map vs simulation, 6 seeds past the last early zero, through `d=12000` | **PASS, 0 mismatches** |

## Result 2 — the observed data, and why it never had power

From simulation (`T=26000`, `d<12000`):

- **Zero words at `d = 2, 7, 28, 399`** only — and correspondingly
  **collision pairs at `d = 0, 5, 26, 397`**, exactly as Lemma A predicts.
- At all four, the predecessor has **even** parity
  (`0xffff`, `0xaaaa`, `0xeeee`, `0xd0d0`; popcounts 16, 8, 12, 6), so
  **doubling never fired**. The conjecture survives the early regime for a
  structural reason: those words are highly regular.
- **The parity regime changes at `d = 403`.** Every word below 403 has even
  parity; from 403 on the split is essentially balanced (**even 6185 / odd
  5815** over `d<12000`), and the words look generic.

That last fact is what undermines the existing evidence. Beyond `d ≈ 403` treat
`w_d` as a generic 16-bit word:

```
P[collision at a given d]  ≈  2^-16  =  1/65536
expected collisions in 403 < d <= 30000  ≈  29600/65536  ≈  0.45
P[zero collisions observed | words generic]  ≈  e^-0.45  ≈  0.64
```

> **Observing "zero exceptions over ~3×10⁴ diagonals" is the single most likely
> outcome even if the conjecture is false.** The test range was a factor ~2
> below the scale at which the triggering event happens at all. The evidence had
> essentially no power.

This is the same failure mode `docs/theory/finite-prefix-counting-bound.md`
identified for the DFAO certificate, in a new guise: a negative was recorded
from a range too small to contain the event that would falsify it. The counting
bound doc is about model-class size; this is about event rate. Both reduce to
*state the scale at which the event can occur before quoting the negative.*

## Result 3 — the first real collision

Iterating the O(1) map past the simulated range (seeded past `d=399`, stopping
at the first zero, so no ambiguous step is ever taken):

```
first consecutive-word collision:  w_53205 = w_53206 = 0x28c3   ->   w_53207 = 0
```

**Six independent seed depths (500, 1000, 2000, 4000, 8000, 11000) agree
exactly.** Expected location under the generic model was `403 + 65536 ≈ 65900`;
observed 53205 — the right order.

`popcount(0x28c3) = 6`, so **parity is even and the period stays 16**. The
conjecture survives its first genuine test — on a coin flip that landed the
right way, not for a reason yet identified.

## Interpretation

The period-16 conjecture is now a *quantified* open question rather than a
well-supported one:

- Doubling requires a collision (**Lemma A**) *and* odd parity (**Lemma B**).
- Collisions past `d ≈ 403` arrive at rate `≈ 2^-16` per diagonal.
- Each doubles with probability `≈ 1/2` if parity is generic.
- **Expected first failure: `d ≈ 403 + 2·65536 ≈ 1.3×10⁵`** — about 4× beyond
  anything simulated, and 2.5× beyond the first collision found here.

So the honest reading is: *if* the settled words are generic beyond the early
structured regime, period-16 is **false**, and fails near `d ~ 10⁵`. If instead
some parity invariant survives past `d = 403`, it is a theorem — but the
even-parity regime demonstrably ends at 403, so no invariant of the simple kind
is available.

Either way the current evidence does not distinguish the two, and the log entry
in `docs/theory/README.md` §3 should not be read as strong support.

Note this does **not** touch the proved results: every left diagonal is
eventually periodic (Theorem, unaffected — periods may simply grow), and the
settled-wedge certificate (276M cells from 29.3 KiB) is unaffected because it
verifies against the actual CA and falls back to simulation at ambiguous
diagonals.

## Next Step

1. **Decide the conjecture.** Walk the map to `d ~ 10⁶`, resolving each zero
   word by a targeted local simulation (only ~15 expected, each a narrow
   diagonal window rather than a full cone). First odd-parity collision refutes
   period-16 outright; reaching `10⁶` with none demands a parity invariant.
2. **Test genericity directly** rather than assuming it: measure the collision
   rate and parity balance of `w_d` over `403 < d < 12000` against a matched
   random-word control, so the `2^-16` model is evidenced rather than asserted.
3. **Do not** extend the "tested to N diagonals, zero exceptions" framing
   without first quoting the event rate and the resulting power.

## Commands

```bash
python experiments/period_doubling.py --pretty \
    --out data/wedge/period_doubling_T26000.json
```

## Artifacts

- `data/wedge/period_doubling_T26000.json` — lemma gates, zero/collision
  inventory, parity split, map-vs-simulation agreement, and the first-collision
  consensus across six seeds.

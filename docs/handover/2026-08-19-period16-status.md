# Handover — The Period-16 Conjecture: Status, Lemmas, and What Is Left

**Date:** 2026-08-19
**Repo:** `D:\APATPROJECTS\rule30-foundry`
**Branch:** `feat/dfao-min-state-curve` (pushed, tracking `origin`)
**Entry points:** `experiments/period_doubling.py`,
`docs/experiment-logs/2026-08-19-period-doubling-criterion.md`,
`docs/CLAIM_LEDGER.md`

---

## TL;DR — RESOLVED: the conjecture is FALSE

> **Period-16 is refuted.** Left diagonal `d = 87867` has minimal period **32**,
> confirmed at two independent simulation sizes. The period is **unbounded**,
> growing as `period(d) ~ 2*log2(d)`.

The trigger was exactly what Lemmas A and B predicted: a collision
`w_87864 = w_87865 = 0x28a8` with **odd** popcount 5, producing a zero word at
`d = 87866` and doubling the period at `d = 87867`. Lemma B is confirmed at
**7 of 7** events across `d < 10^6`, including the two even-parity collisions
that correctly kept the period at 16.

Full detail: [`../experiment-logs/2026-08-19-period16-refuted.md`](../experiment-logs/2026-08-19-period16-refuted.md).

Period histogram over `d < 10^6` (all 10^6 diagonals settled, T = 1476886):

```
{1: 16, 2: 10, 4: 56, 8: 668, 16: 87136, 32: 912114, 64: 0}
```

Nothing proved was lost. **"Every left diagonal is eventually periodic" remains
a Theorem** (periods simply grow, as the propagation lemma always allowed), and
the 276M-cell wedge Certificate stands — it was run entirely inside the
period-16 region. Its `O(1)` 16-bit pattern map is valid only for `d < 87866`.

---

## Background in one paragraph

Each settled left diagonal `d` is a 16-bit word `w_d` phase-locked to
`t mod 16`. The diagonal recursion
`D_d(t+1) = D_{d-2}(t) XOR (D_{d-1}(t) OR D_d(t))` lifts to a pattern map
`(w_{d-2}, w_{d-1}) -> w_d` (`pattern_map_step` in
`experiments/diagonal_recursion.py`), well defined whenever `w_{d-1} != 0`. The
period-propagation lemma has a doubling branch that fires exactly when
`w_{d-1} = 0`. The conjecture is that the period always divides 16, i.e. that
branch never actually doubles anything.

---

## Result 1 — the two lemmas (Theorem, in the ledger)

**Lemma A (collision criterion).** For `v != 0`,
`pattern_map_step(u, v) = 0` **iff** `u = v`.

> *Proof.* `w = 0` forces `0 = w[t+1] = u[t] XOR (v[t] OR 0) = u[t] XOR v[t]`
> for every `t`, i.e. `u = v`. Conversely if `u = v` then `w = 0` satisfies the
> recursion, and since `v != 0` the one-period composite is constant, so that
> solution is the unique eventual one. []

Hence **`w_d = 0` iff `w_{d-2} = w_{d-1}`**: the doubling branch fires exactly
at a *collision between consecutive settled words*.

**Lemma B (doubling criterion).** At a collision the composite is
`x -> x XOR c` with `c = parity(w_{d-2})`. The period doubles iff `c = 1`; if
`c = 0` it stays. []

> **Consequence.** Period-16 holds through `D` **iff every consecutive-word
> collision below `D` has an even-parity predecessor.** Both conditions are
> checkable without simulating past the seed.

Verification gates (all in `experiments/period_doubling.py`, 3.4 s, exits
non-zero on failure):

| gate | result |
|---|---|
| Lemma A exhaustive at period 8 (all 65280 pairs with `v != 0`) | 0 violations |
| Lemma A sampled at period 16 (200k pairs) | 0 violations |
| Lemma A against simulation, `d = 0..11999` | 0 violations |
| Map vs simulation, 6 seeds, through `d = 12000` | PASS, 0 mismatches |

---

## Result 2 — the d=399 zero-word bug (read this before touching the map)

**This bug produced a wrong answer that survived one round of self-checking. Do
not re-introduce it.**

**Symptom.** A first collision hunt seeded the pattern-map iteration at
`d = 512` and reported the first collision at `d = 53205/53206`. A robustness
sweep over seed depths then returned **contradictory** answers:

| seed_d | reported first collision |
|---|---|
| 256, 300 | `(397, 398)`, continuing to `d = 53456` |
| 512, 700, 1024 | `(53205, 53206)` |

**Root cause.** There is a genuine zero word at `d = 399`, caused by the real
collision `w_397 = w_398 = 0xd0d0`. The naive map iteration **cannot step
through a zero word**: when `v = 0` the composite is affine, the next word is
ambiguous, and recovering it needs the actual transient from simulation. (This
is precisely the "fallbacks to simulation: 1" / "4 ambiguous" counters already
present in the wedge certificate — the information was there and was not used.)
Low seeds ran *into* the zero and mis-stepped past it; high seeds happened to be
seeded *beyond* it from simulation and were correct by luck.

**Fix, as encoded in `period_doubling.py`:**

1. Compute **all** zero words from simulation first.
2. Seed the map iteration only strictly past the last zero word:
   `seeds = [s for s in (500, 1000, 2000, 4000, 8000, 11000) if last_zero < s < diagonals]`.
3. **Stop at** the first zero word rather than attempting to continue through it.

After the fix all six seeds agree exactly and the map matches simulation with 0
mismatches through `d = 12000`.

**Lesson for the next agent:** seed-depth agreement is the cheap test that
catches this class of bug. Any iteration of the pattern map must be run from
several independent seeds and required to agree, not just checked once.

---

## Result 3 — the observed data, and why it had no power

From simulation at `T = 26000`, `d < 12000`:

- **Zero words occur only at `d = 2, 7, 28, 399`**, with collision pairs at
  `d = 0, 5, 26, 397` — exactly as Lemma A predicts.
- ~~All four predecessors have **even** parity, so doubling never fired.~~
  **WRONG — corrected 2026-08-19.** That reads the popcount off a 16-bit
  *padded* word; a period-`p` word (`p < 16`) is stored as `16/p` copies, which
  doubles its popcount and forces even parity. At their **minimal** periods all
  four predecessors are **odd** and the period doubled at every one:

  | zero `d` | period | predecessor at that period | popcount | parity |
  |---|---|---|---|---|
  | 2 | 1 → 2 | `0x1` | 1 | odd |
  | 7 | 2 → 4 | `0x2` | 1 | odd |
  | 28 | 4 → 8 | `0xe` | 3 | odd |
  | 399 | 8 → 16 | `0xd0` | 3 | odd |

  This is *how the period reached 16*. Fixed in `experiments/period_doubling.py`
  (`minimal_period_of_word`).
- **The parity regime ends at `d = 403`.** Every word below 403 has even parity;
  from 403 on the split is essentially balanced (**even 6185 / odd 5815** over
  `d < 12000`) and the words look generic.

Treating `w_d` as a generic 16-bit word past `d ~ 403`:

```
P[collision at a given d]                    ~  2^-16  =  1/65536
expected collisions in 403 < d <= 30000      ~  29600/65536  ~  0.45
P[zero collisions observed | words generic]  ~  e^-0.45      ~  0.64
```

> **Observing "zero exceptions over ~3x10^4 diagonals" is the single most likely
> outcome even if the conjecture is false.** The evidence had essentially no
> power.

This is the same failure mode as the retracted DFAO certificate
(`docs/theory/finite-prefix-counting-bound.md`), in a new guise: a negative
quoted from a range too small to contain the event that would falsify it. That
doc is about model-class size; this is about event rate. Both reduce to *state
the scale at which the event can occur before quoting the negative.*

---

## Result 4 — the first real collision

Iterating the O(1) map past the simulated range (seeded past `d = 399`, stopping
at the first zero, so no ambiguous step is ever taken):

```
first consecutive-word collision:  w_53205 = w_53206 = 0x28c3   ->   w_53207 = 0
```

**Six independent seed depths (500, 1000, 2000, 4000, 8000, 11000) agree
exactly.** Expected location under the generic model was `403 + 65536 ~ 65900`;
observed 53205 — the right order of magnitude.

`popcount(0x28c3) = 6`, so **parity is even and the period stays 16**. The
conjecture survives its first genuine test — on a coin flip that landed the
right way, not for a reason yet identified.

**Expected first failure: `d ~ 403 + 2*65536 ~ 1.3x10^5`** — about 4x beyond
anything simulated, and 2.5x beyond the first collision found here.

---

## Where the ledger now stands

| Claim | Level |
|---|---|
| Doubling fires exactly at a consecutive-word collision, and doubles iff the predecessor parity is odd (Lemmas A/B). | **Theorem** (new) |
| Every left diagonal is eventually periodic with period dividing 16. | **Proof candidate — evidence downgraded 2026-08-19** |
| Every left diagonal is eventually periodic. | Theorem (**unaffected** — periods may simply grow) |
| Settled wedge: 276,326,150 cells from 29.3 KiB, 0 mismatches. | Certificate (**unaffected** — it verifies against the actual CA and falls back at ambiguous diagonals) |

---

## Status of the tasks that were pending

| Task | Status |
|---|---|
| 1. Regression test for zero-word handling | **Done** — `experiments/zero_word_regression.py`, 5 gates, mutation-tested |
| 2. Justify the `2^-16` independence assumption | **Done** — measured to 0.5% over 1.39e9 pairs, `2026-08-19-settled-word-genericity.md` |
| 3. Walk to `d ~ 10^6` | **Done, and it refuted the conjecture** — `2026-08-19-period16-refuted.md` |

### Now open

1. **Generalize the wedge pattern map** to `2*log2(d)`-bit words and re-issue the
   compression certificate without the period-16 assumption.
2. **Confirm the growth law** at the predicted `32 -> 64` event near
   `d ~ 8.6x10^9`. Direct simulation is out of reach (`T ~ 1.2x10^10`), so this
   needs the generalized map first.
3. **DFAO minimal-state curve** — still has unfilled placeholders; PySAT now works.

## Routes already closed — do not re-propose

| Route | Why it fails |
|---|---|
| Floyd cycle certificate | The state map is **partial**. At a zero word two words satisfy the recursion and only the diagonal transient (length `~1.34d`, unbounded) picks one. Measured `P[cycle] ~ 10^-14231`. |
| Exhaustive `2^32` reachability bitmap | Same flaw — assumes a total map. |
| Any **bounded**-period claim | The period is unbounded: `~ 2*log2(d)`. |

### The recurring failure mode — read this before quoting any negative

Three separate errors in this thread had the same shape: *a negative quoted from
a range, or a representation, too small to contain the falsifying event.*

1. The retracted DFAO certificate — model class too small (counting bound).
2. The original period-16 evidence — test range below the event rate (power).
3. The 16-bit word representation — **could not express a period-32 diagonal at
   all**, so the counterexample was invisible and surfaced only as an
   unexplained drop in the "settled" count.

Ask not only whether the range is large enough, but whether the representation
can **express the counterexample**.

### Standing rule
**Do not** extend the "tested to N diagonals, zero exceptions" framing without
first quoting the event rate and the resulting power.

## Reproduction

```bash
python experiments/period_doubling.py --pretty \
    --out data/wedge/period_doubling_T26000.json
```

Runs in ~3.4 s, exits non-zero on any gate failure. Expected output:

```
Lemma A  : exhaustive period-8 0 violations, sampled 0 violations
Lemma A on simulation: 0 violations over 12000 settled diagonals
zero words      : [2, 7, 28, 399]
collision pairs : [0, 5, 26, 397]
doubling fired  : False
parity split    : even 6185 / odd 5815, first odd at d=403
map vs sim      : PASS (6 seeds, 0 mismatches)
first collision : d=53205/53206 word=0x28c3 parity=0 -> stays 16
seeds agree     : True
```

Artifact: `data/wedge/period_doubling_T26000.json`.

---

## Environment note (restored 2026-08-19)

Python deps were missing on this machine (likely lost in the July 2026 SSD
migration) and are now installed: numpy 2.5.2, scipy 1.18.0, **python-sat**
(in-process CaDiCaL/Glucose/Minisat — this clears the blocker recorded in
`docs/experiment-logs/2026-06-15-prize-frontier.md`, *"No local kissat, cadical,
minisat, or glucose executable was found"*), and cupy 14.1.1 with CUDA headers
(`cupy-cuda12x[ctk]`; JIT compiles on the GTX 1060, and the `CUDA_PATH` warning
is benign).

## Also open, unrelated to this thread

- **DFAO minimal-state curve** — `docs/experiment-logs/2026-08-15-dfao-min-state-curve.md`
  is committed with unfilled placeholders. The 7 random controls exist
  (`data/prize/2026-08-15-random-seeds/`, `n = 8..48`, `s* ~ 3.4 -> 12.3`); only
  the Rule 30 and Thue-Morse arms are missing. Cheap, and PySAT now works.
- **5 open agent PRs (#12-#16)** — deliberately untouched, awaiting triage.

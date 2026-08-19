# Theory Gate

`AGENTS.md` tells every agent to read this file before proposing theory-motivated
work. Until 2026-08-15 it did not exist, and that gap is directly implicated in
the vacuous DFAO certificate documented below. Keep it current.

The job of this file is to stop three recurring failure modes:

1. Running a search whose negative outcome was guaranteed by counting.
2. Measuring an *ensemble* property and reporting it as progress on a prize.
3. Re-deriving something the rule's algebra already gives for free.

---

## 0. The prize objects (and a numbering correction)

Wolfram's three Rule 30 prizes, in the official numbering:

| # | Question |
|---|---|
| 1 | Does the center column always remain non-periodic? |
| 2 | Does each color occur on average equally often in the center column? |
| 3 | Does computing the nth cell of the center column require at least O(n) computational effort? |

> **Correction:** several existing logs use a swapped numbering — Problems 2 and 3
> are interchanged in `docs/problem-statements/center-column-shortcuts.md`
> ("prize problem 2" for the shortcut question), in `S_linear_complexity.md`
> ("Prize-2 probe" for an LFSR shortcut), and in `R_left_edge_cone.md`
> ("Prize-3 equidistribution"). The shortcut/effort question is **Problem 3**;
> equidistribution is **Problem 2**. Use the table above.

**All three prizes concern one single deterministic orbit** — the seed
`…0001000…`. There is no ensemble. This has a sharp consequence:

> A quantity defined by averaging over random initial conditions, or by
> perturbing an initial condition, is a property of *the rule*, not of *the prize
> object*. It can motivate work on the seed orbit; it cannot itself be progress.

This is why `Q_damage_velocity.md` and `R_left_edge_cone.md`, though correct,
close with an explicit ceiling note. Respect it. If a proposed experiment needs
an ensemble to be defined, ask what its single-seed analogue is and measure that
instead.

---

## 1. What the algebra gives for free (do not re-measure these)

Rule 30 in algebraic normal form:

```
a(t+1, i)  =  a(t,i-1)  XOR  ( a(t,i) OR a(t,i+1) )
           =  a(t,i-1) ⊕ a(t,i) ⊕ a(t,i+1) ⊕ a(t,i)·a(t,i+1)
```

**Left-permutivity.** `a(t,i-1)` enters as a pure XOR, so for any fixed
`(a(t,i), a(t,i+1))` the map `a(t,i-1) ↦ a(t+1,i)` is a bijection. Consequences,
all theorems, none needing measurement:

- **Rightward damage speed is exactly 1** on every background. `v_right = 1` is
  not an empirical finding.
- **The uniform Bernoulli(1/2) measure is invariant.** A random i.i.d. fair row
  stays i.i.d. fair forever, so the center column is provably 50/50 *for random
  ICs*. Prize 2 is therefore trivial for random ICs and its entire difficulty is
  concentrated in the single seed. Numerical "the column looks 50/50" runs on
  random ICs measure nothing.
- **Sideways determinism.** Rearranging,
  `a(t,i-1) = a(t+1,i) ⊕ ( a(t,i) OR a(t,i+1) )`, so any two adjacent columns
  determine the entire half-plane to their left. Leftward reconstruction is
  deterministic; rightward reconstruction branches exactly where `a(t,i)=1`.

`a(t,i+1)` sits inside the OR, so leftward propagation is *conditional* (gated on
`a(t,i)=0`). Leftward speed is the only empirically interesting one.

---

## 2. The finite-prefix counting bound

See **[finite-prefix-counting-bound.md](finite-prefix-counting-bound.md)**.

One-line form: searching a model class `M` against `n` bits and finding no fit is
informative **only if `log2|M| >= n`**. Otherwise every sequence gives that
answer.

Standing consequences:

- The recorded certificate "no 1-5 state DFAO fits the first 128 center bits" is
  **vacuous by a factor of `2^-100`**. Do not extend it to more states at n=128;
  states 6-8 are still vacuous by `2^-72`.
- Experiment S (linear complexity `L(n) = n/2`) is **sound** — it sits exactly at
  the counting threshold, which is also the maximum. It is the design template.
- Always report a **curve against a null**, never a **point against nothing**.

---

## 3. Structural facts measured on the seed orbit

### The left-diagonal recursion (2026-08-15) — THEOREM

Define left diagonal `d` as `D_d(t) = a(t, -t+d)`, the cell at fixed offset `d`
inward from the left edge of the light cone. Substituting `i = -t+d-1` into the
Rule 30 update gives a **closed recursion on the diagonals alone**:

```
D_d(t+1)  =  D_{d-2}(t)  XOR  ( D_{d-1}(t)  OR  D_d(t) )
```

**Lemma (period propagation).** If `D_{d-2}`, `D_{d-1}` are eventually periodic
with common period `p`, then `D_d` is eventually periodic with period dividing
`2p`. *Proof:* `φ_t(x) = D_{d-2}(t) ⊕ (D_{d-1}(t) ∨ x)` is constant when
`D_{d-1}(t)=1` and the bijection `x ↦ D_{d-2}(t) ⊕ x` when `D_{d-1}(t)=0`, so the
one-period composite is either constant (⇒ period `p`) or `x ↦ x ⊕ c` (⇒ `p` or
`2p`). ∎

**Corollary.** `D_0 ≡ 1` and `D_1 ≡ 1` (t≥1) come straight from the rule at the
cone edge, so by induction **every left diagonal is eventually periodic.**

**Conjecture (period-16), open.** Every left diagonal is eventually periodic with
period **dividing 16**. Verified over `d = 0..11997` at `T = 40000`: max period
16, zero exceptions. Proving it requires bounding how often the lemma's
period-doubling branch (predecessor eventually all-zero over a full period) can
fire. Verifier: `python experiments/diagonal_recursion.py --pretty`.

> **Do not describe the cone as "periodic wedge + aperiodic core".** Every
> diagonal is eventually periodic; only the *settling time* varies. The right
> decomposition is **settled vs not-yet-settled at the horizon**.

### The settled wedge (2026-08-15) — measurement

Measured settling law, single seed, `T = 65536`, `W = 49152`, 44906 diagonals:

```
settle(d)  ≈  1.3389 · d          (subrange spread 0.023)
```

Independently replicated at `s = 1.3411` by a second agent from spec, through the
repo's own verified kernels, all correctness gates PASS — agreement to 0.2%.

Derived constants:

- Settled wedge occupies a fraction `1/(2s) ≈ 0.373` of the light-cone area.
- Its inner boundary moves leftward at speed `1 - 1/s ≈ 0.253`.
- The wedge admits an **`O(t)`-size description with `O(1)` random access**, via
  the pattern map below — an asymptotic win on that region, not a constant factor.
- Verified by a falsifiable prediction: requiring periodicity only on a trailing
  window `[αT, T]` yields first-failure depth `≈ αT/s`, confirmed to within 0.4%
  for α ∈ {0.25, 0.5, 0.75, 0.9, 0.95}.
- **Cap-insensitive**: identical answers with period caps 16, 64, 256, 1024, so
  the wedge boundary is a sharp transition, not a period exceeding the search cap.
- **Clean entropy dichotomy**: pre-settle block-entropy rate ≥0.9975 and zlib
  ratio ≥1.0059 (matched random control: 0.99961 / 1.0044); post-settle zlib
  ratio ≤0.0238. No intermediate regime.

### The pattern map (2026-08-15)

Each settled diagonal's eventual behaviour is a 16-bit word `w_d` phase-locked to
`t mod 16`. Whenever `w_{d-1}` is not identically zero the one-period composite is
constant, so `(w_{d-2}, w_{d-1}) -> w_d` is a well-defined **O(1) map on 16-bit
words**. Verified over `d = 2..11998`: 11993 agree, 0 mismatch, 4 ambiguous
(exactly the all-zero cases the lemma flags).

Consequence: `w_0 ... w_D` are computable in `O(D)` total, and any settled cell is
`a(t, -t+d) = w_d[t mod 16]`. **The settled wedge - `Theta(t^2)` cells - has an
`O(t)` description with `O(1)` random access.**

**Certified.** Seeding `w_0..w_255` from a short simulation and generating the
rest by the map reproduces **276,326,150** settled cells with **zero mismatches**
from a 29.3 KiB description - a 1151x compression. One command, ~24 s:
`python experiments/diagonal_recursion.py --steps 30000 --diagonals 15000 --pretty`.

**A proof route that fails, recorded so nobody repeats it.** The pair space is
finite (`2^32`), so `d -> (w_{d-1}, w_d)` must eventually cycle, and exhibiting
the cycle would prove period-16. Over 11999 settled diagonals there are 11999
**distinct** pairs and 10827 distinct patterns - no repeat, no sign of one. The
pattern sequence is itself Rule-30-like in `d`. Do not spend compute here.

**Why this is prize-facing (Problem 3).** It is a single-seed quantity, so it is
not subject to the ensemble ceiling in section 0. The cone splits into a settled
wedge (~37%, `O(t)` description, `O(1)` access) and an unsettled core (~63%, no
description found). The settled region is asymptotically free - and irrelevant,
because

```
centre bit at time T   <->   diagonal d = T,   settle(T) ~ 1.34*T  >  T
```

so the **centre column is permanently in the unsettled region at every horizon.**
The left-edge structure is not merely insufficient for Problem 3, it is *disjoint
from the prize object*. Any sub-quadratic centre-column algorithm must compress
the unsettled core, which is indistinguishable from i.i.d. fair coin flips under
periodicity, block entropy, and zlib against a matched control.

"The centre column never settles" is the Problem-1 statement read along diagonals
instead of down the column.

Note the numerical coincidence with the ensemble damage velocity
`λ_L ≈ 0.244` from `R_left_edge_cone.md`. These are different quantities
(structural vs interventional) and the agreement is suggestive, not established.

---

## 4. Routes currently closed

| Route | Status | Evidence |
|---|---|---|
| GF(2) linear recurrence / LFSR shortcut | **Closed** — maximal linear complexity `L(n)=n/2` | `S_linear_complexity.md` |
| b=2 coarse-grain reducibility | **Closed** — generic chaotic leakage, not Rule-30-specific | `2026-06-13-coarse-grain-same-statistics-null.md` |
| b=3 coarse-grain, r=1, tested shears | **Closed** | `2026-06-14-b3-coarse-grain-verdict.md` |
| Aggregate randomness tests (A-L) | **Exhausted** — no promotion path | `idea-bank/theoretical-reframe-2026-03-28.md` |
| Small-DFAO negatives at large `n` | **Vacuous by construction** — do not extend | §2 above |
| Diagonal-periodicity compression of the cone | **Closed** — the settled 37% is `O(t)`-describable but *disjoint from the centre column* | §3 above |
| Finite-orbit proof of the period-16 conjecture | **Closed** — pair orbit shows 11999/11999 distinct, no cycle within reach | §3 above |
| "Every left diagonal is eventually periodic" as a shortcut | **Closed** — settling time grows linearly, centre column never settles | §3 above |

---

## 5. Standing rules for new proposals

Before proposing an experiment, answer in one line each:

1. **Prize object?** Does this act on the single seed orbit, or on an ensemble?
   If an ensemble, what is the single-seed analogue?
2. **Counting check?** State `log2|M|` and `n`. If `log2|M| - n` is very
   negative, do not run it.
3. **Null and positive control?** What sequence proves the method can detect a
   shortcut that really is there (e.g. Thue-Morse for automaticity)? What is the
   matched random baseline?
4. **Promotion path?** What would turn the result into a checkable artifact — a
   verifier command, an UNSAT certificate, a lemma? If there is none, the
   experiment is probably not worth running.

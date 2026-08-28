# Experiment Log — Structured/Chaotic Decomposition of the Seed Light Cone

- Date: 2026-08-15
- Title: Settled wedge vs unsettled core of the single-seed Rule 30 cone
- Prize: **Problem 3** (computational effort). See the numbering correction in
  `docs/theory/README.md` §0 — several older logs swap Problems 2 and 3.
- Naming: dated filename rather than a frontier letter, per `AGENTS.md`
  ("do not assume the next experiment is the next letter unless the canonical
  M/N/O/P mapping has been reconciled"). Follows the Q → R → S thread.

## Goal

`R_left_edge_cone.md` closed with an explicit ceiling: damage-velocity work is
defined over an *ensemble* of perturbed initial conditions, and all three prizes
concern a **single deterministic orbit**, so that thread "cannot, by itself,
touch the prize object."

This experiment moves the same geometric question onto the prize object. It asks
a Problem-3 question with no ensemble anywhere in it:

> How much of the Rule 30 light cone from the single black cell has settled into
> structure that could be replaced by a closed form, and how much has not?

The answer bounds what *any* shortcut built on the rule's visible regularity can
possibly buy.

## Setup

Rule 30, single black cell, open boundaries, bit index increasing right:

```
a(t+1, i) = a(t, i-1) XOR ( a(t, i) OR a(t, i+1) )
```

Define **left diagonal `d`** as `D_d(t) = a(t, -t+d)` — the cell at fixed offset
`d` inward from the left edge of the light cone. This is the right object to
study: a *fixed column* `a(t, x)` has relative position `x/t → 0`, so every fixed
column ends up in the chaotic centre. Only edge-parallel diagonals can carry the
edge's structure.

Key observation driving the design: a diagonal is **not** periodic from the
start. Diagonal `d` is born near the chaotic centre at `t ≈ d` and later
*settles* into eventual periodicity as the structured wedge overtakes it. So the
quantity to measure is a settling time, not a break time:

```
settle(d) = last t at which D_d(t) != D_d(t-p),  minimised over candidate p
```

after which `D_d` is `p`-periodic for the best `p`.

- Engine: exact Python big-integer Rule 30 (whole row as one bignum,
  `((s<<1) ^ (s | (s>>1))) & mask`), analysis vectorised in numpy over a
  bit-packed `(T+1, W/8)` buffer. CPU only — cupy is not installed on this
  machine after the July 2026 SSD migration.
- Scale: `T = 65536`, `W = 49152` diagonals (~400 MB), plus a scaling series at
  `T = 4096 … 65536`.

## Method

```bash
python experiments/wedge_profile.py --steps 65536 --diagonals 49152 --pretty \
    --out data/wedge/wedge_profile_T65536.json
python experiments/wedge_profile.py --steps 8192 --diagonals 6144 --pretty   # quick
```

## Verification against a trusted reference

- **Ground-truth gate (hard failure).** The centre column prefix must equal
  `1101110011000101` = **OEIS A051023**. This is an external check, not a
  self-consistency check.
- **Indexing gate (hard failure).** Diagonals 0 and 1 are provably identically 1
  for all `t ≥ 1` (the two leftmost live cells of the cone follow directly from
  the rule). The script aborts if the extracted diagonals 0/1 are not all-ones,
  which catches any off-by-one or bit-order error in the window extraction.
- **Independent replication** by a second agent, written from a spec without
  sight of this code, using the repo's own verified kernels in
  `experiments/rule30_open_utils.py`. Its own gates — `packed_vs_naive`,
  `center_column_A051023`, `frame_vs_repo_spacetime` (1.83M cells), repo
  self-tests — all PASS. See `experiments/wedge_verify.py`.
- **Closed-form cross-check.** The diagonal recursion in Observation 0 is an
  algebraic identity, verified bit-exact independently of the wedge measurement
  by `experiments/diagonal_recursion.py`.

## Controls

- **Period-cap invariance.** The whole measurement is repeated with candidate
  period caps 16, 64, 256, 1024. If the wedge boundary were an artifact of the
  search cap, the depth would grow with the cap.
- **Random control for the entropy split.** A seeded i.i.d. fair sequence
  (`seed=30`) is run through the identical entropy and compression estimators.
- **Falsifiable prediction.** The linear-settling model predicts that requiring
  periodicity only on a trailing window `[αT, T]` gives first-failure depth
  `αT/s`. Tested at α ∈ {0.25, 0.5, 0.75, 0.90, 0.95, 0.98}.

## Observations

### 0. The diagonals obey a closed recursion — and this is a theorem, not a fit

Substituting `i = -t+d-1` into the Rule 30 update gives, exactly:

```
D_d(t+1)  =  D_{d-2}(t)  XOR  ( D_{d-1}(t)  OR  D_d(t) )
```

a **closed recursion on the left diagonals alone**. Verified bit-exact over
`d = 2..2999`, `t = 0..19999` (PASS, no mismatches).

**Lemma (period propagation).** If `D_{d-2}` and `D_{d-1}` are eventually
periodic with common period `p`, then `D_d` is eventually periodic with period
dividing `2p`.

*Proof.* For `t` past the settling time define `φ_t(x) = D_{d-2}(t) ⊕ (D_{d-1}(t) ∨ x)`,
so `D_d(t+1) = φ_t(D_d(t))`. If `D_{d-1}(t) = 1` then `φ_t` is the constant map
`x ↦ D_{d-2}(t) ⊕ 1`; if `D_{d-1}(t) = 0` then `φ_t` is the bijection
`x ↦ D_{d-2}(t) ⊕ x`. The one-period composite `Φ = φ_{T₀+p-1} ∘ … ∘ φ_{T₀}` is
therefore either constant or of the form `x ↦ x ⊕ c`. If any `t` in the period
has `D_{d-1}(t) = 1`, `Φ` is constant, so `D_d(T₀+kp)` is the same for every
`k ≥ 1` and `D_d` is eventually `p`-periodic. Otherwise `D_{d-1} ≡ 0` on the
period and `Φ(x) = x ⊕ c` with `c = ⊕_t D_{d-2}(t)`, giving period `p` if `c=0`
and `2p` if `c=1`. ∎

**Corollary.** The base cases `D_0 ≡ 1` (t≥0) and `D_1 ≡ 1` (t≥1) follow directly
from the rule at the cone edge. By induction, **every left diagonal is eventually
periodic.** The period at most doubles per diagonal, and only at a diagonal whose
predecessor is eventually all-zero across a full period.

Worked start, matching measurement exactly: `D_2 ≡ 0` (period 1),
`D_3(t+1) = 1 ⊕ D_3(t)` so period 2, `D_4(t+1) = D_3(t) ∨ D_4(t)` which absorbs
to 1 so period 1 — reproducing the measured periods 1,1,1,2,1,… at d = 0..4.

**Conjecture (period-16).** Every left diagonal is eventually periodic with
period **dividing 16**. Tested over `d = 0..2999` at `T = 20000`: histogram
`{1:10, 2:8, 4:37, 8:356, 16:2589}`, **maximum 16, zero exceptions**. The
independent replication saw the same ceiling over `d = 768..29491`
(`{4:2, 8:116, 16:28606}`).

### Framing correction this forces

The cone is **not** "periodic wedge + aperiodic core". Every diagonal is
eventually periodic; what varies is *when*. The correct decomposition is
**settled vs not-yet-settled at horizon `T`**, with `settle(d) ≈ 1.34·d`. The
observations below should be read that way — "aperiodic core" throughout means
"not yet settled at the horizon", which is empirically full-entropy over its
pre-settling window.

### 1. Settling is linear

```
settle(d) = 1.3389 · d − 62      (T=65536, W=49152, 44906 fitted diagonals)
```

Stable across disjoint quarters of the `d` range (subrange spread 0.023).

**Independent replication** (separate agent, written from spec, using the repo's
own verified kernels in `experiments/rule30_open_utils.py`, no sight of this
code) obtained `s = 1.3411` through the origin / `1.3457` with intercept, over
28724 diagonals, with `packed_vs_naive`, `A051023`, and `frame_vs_repo_spacetime`
(1.83M cells) all PASS. Agreement to 0.2%.

### 2. The prediction holds

| α | ratio measured/predicted |
|---|---|
| 0.25 | 1.0039 |
| 0.50 | 0.9996 |
| 0.75 | 1.0019 |
| 0.90 | 1.0016 |
| 0.95 | 1.0007 |

Confirmed to within 0.4% over a 4× range of window fractions. The replication
independently got 1.0049 at α = 0.25.

### 3. The boundary is sharp, not a search artifact

Period caps 16 / 64 / 256 / 1024 give **identical** depths at every tested `T`.
Diagonals in the wedge have genuinely tiny periods (powers of two, all observed
≤ 16); one diagonal further in, nothing periodic exists at any tested scale.

### 4. The entropy split is clean and binary

Splitting each diagonal's history at `settle(d)`:

| phase | block-entropy rate | zlib ratio |
|---|---|---|
| pre-settle (unsettled) | min 0.9975 | min 1.0059 (**incompressible**) |
| post-settle (wedge) | — | max 0.0238 (**compresses to <2.4%**) |
| i.i.d. random control | 0.99961 | 1.0044 |

The pre-settle phase is statistically indistinguishable from the random control.
**There is no intermediate, partially-compressible regime.** Every cell of the
cone is either in a diagonal that is eventually periodic with period ≤ 16, or in
one that behaves like fair coin flips.

### 5. Scaling of the constant (honest uncertainty)

| T | slope | subrange spread | wedge fraction | boundary speed |
|---|---|---|---|---|
| 4096 | 1.32121 | 0.068 | 0.3784 | 0.2431 |
| 8192 | 1.33341 | 0.045 | 0.3750 | 0.2500 |
| 16384 | 1.33828 | 0.025 | 0.3736 | 0.2528 |
| 65536 | 1.33890 | 0.023 | 0.3734 | 0.2531 |

The slope **drifts upward** with `T` while the subrange spread shrinks. The
exactly-4/3 value at `T = 8192` (giving wedge = 3/8, boundary speed = 1/4) is
therefore a finite-size coincidence the drift passes through, **not** an exact
constant. Do not quote 1/4 as exact.

## Result

Asymptotic decomposition of the single-seed light cone:

- **settled wedge** — area fraction `1/(2s) ≈ 0.373 ± 0.004`
- **not-yet-settled core** — area fraction `≈ 0.627`
- wedge inner boundary moves leftward at speed `1 − 1/s ≈ 0.253 ± 0.005`
- settled wedge has an `O(t)`-size description with `O(1)` random access
  (Observations 6-7; supersedes the `1.60×` figure) — **certified** at
  276,326,150 cells reproduced from 29.3 KiB with zero mismatches
- the centre column is **never** in the settled region: `settle(T) ≈ 1.34·T > T`

Canonical artifact: `data/wedge/wedge_profile_T65536.json` (545 s, CPU).
Recursion/period verifier: `data/wedge/diagonal_recursion_T40000.json`.

### 6. The pattern map — and a self-correction

Attempting to *prove* the period-16 conjecture produced a result that revises
the "1.60×" framing above. It is left in place so the correction is legible.

Each settled diagonal's eventual behaviour is a 16-bit word `w_d` (phase-locked
to `t mod 16`). The recursion lifts to these words: whenever `w_{d-1} ≢ 0`, the
one-period composite is constant, so

```
(w_{d-2}, w_{d-1})  ↦  w_d          is a well-defined, O(1) map on 16-bit words
```

Verified against measurement over `d = 2..11998`: **11993 agree, 0 mismatch, 4
ambiguous** (the ambiguous cases are exactly the `w ≡ 0` diagonals the lemma
flags).

**The proof route I expected to work does not.** The state space of pairs is
finite (`2^32`), so the orbit `d ↦ (w_{d-1}, w_d)` must eventually cycle, and
exhibiting that cycle would prove period-16. It does not cycle: over 11999
settled diagonals there are **11999 distinct pairs** — every one unique — and
10827 distinct 16-patterns. The pattern sequence is itself Rule-30-like in the
`d` direction. A cycle could take `~2^32` steps; this route is not reachable.

**What this changes.** The settled wedge is *far* more compressible than the
1.60× figure implies. The words `w_0 … w_D` are generated in `O(D)` total time by
the O(1) map, and for any cell with `t > settle(d)` we have
`a(t, -t+d) = w_d[t mod 16]`. So:

> The settled wedge — `Θ(t²)` cells, ≈37% of the cone — admits an `O(t)`-size
> description with `O(1)` random access.

That is an asymptotic win on that region, not a 1.60× constant. **Strike the
1.60× ceiling; it was too weak, for the wrong reason.**

Note the words are *not* incompressible: they have a short generating program.
Observation 4's incompressibility claim applies to the **pre-settle** phase only,
which is correct as stated but must not be read as covering the wedge.

### 7. Certificate — the settled wedge reproduced from 29 KiB

Observation 6's claim is now a mechanically checkable artifact rather than an
argument. Procedure: seed `w_0 … w_255` from a short simulation (this absorbs the
finitely many ambiguous all-zero diagonals), generate every later word with the
O(1) pattern map, then predict

```
a(t, -t+d)  =  w_d[t mod 16]      for every cell with t > 1.45·d + 200
```

and compare against the actual CA.

| quantity | value |
|---|---|
| words generated by the map | 14743 |
| fallbacks to simulation | 1 |
| settled-region cells checked | **276,326,150** |
| **mismatches** | **0** |
| description size | 15000 words × 16 bits = 240000 bits (29.3 KiB) |
| compression ratio | **1151×** |

Reproduce in ~24 s:

```bash
python experiments/diagonal_recursion.py --steps 30000 --diagonals 15000 \
    --tail 4096 --pretty --out data/wedge/diagonal_recursion_T30000.json
```

This is a **Certificate** in the ledger's sense: a finite artifact another agent
can verify with one command, and the script exits non-zero if any of the recursion
check, base cases, or generator check fails.

## Interpretation

**This is a quantified obstruction for Problem 3 — and the sharper version
survives the correction above.**

The cone decomposes as:

| region | share | description cost | random access |
|---|---|---|---|
| settled wedge | ≈37% | `O(t)` bits via the pattern map (certified, 1151× at T=30000) | `O(1)` |
| unsettled core | ≈63% | none found | — |

The settled region is asymptotically free. **It does not matter**, because of
where the centre column sits:

```
centre bit at time T  ↔  diagonal d = T,  settle(T) ≈ 1.34·T  >  T
```

The centre column is **permanently in the unsettled region** — it never settles,
at any horizon. So the entire `O(t)` shortcut applies to a part of the cone the
prize question never touches, and the cost of computing the centre bit stays
`Θ(T²)`.

That is a cleaner obstruction than a constant-factor bound: the left-edge
structure is not merely *insufficient*, it is **disjoint from the prize object**.
Any sub-quadratic centre-column algorithm must compress the unsettled core, and
observation 4 says that region is indistinguishable from i.i.d. fair coin flips
under periodicity, block entropy, and general-purpose compression.

Seen this way, "the centre column never settles" is the Problem-1
(non-periodicity) statement viewed along diagonals rather than down the column.

Observation 6 answers the phase question this paragraph originally left open:
the phase is carried by the 16-bit word `w_d`, and the pattern map supplies it in
`O(1)` per diagonal. The wedge is cheaper than first claimed, and still
irrelevant to the prize.

**Relation to `λ_L`.** The boundary speed `≈ 0.253` sits next to the ensemble
leftward damage velocity `λ_L ≈ 0.244` from `R_left_edge_cone.md`. These are
different quantities — one structural on a single orbit, one interventional over
an ensemble — and this experiment does **not** establish that they are the same
constant. The proximity is suggestive and worth a dedicated test; it is not a
result. Note also that the structural estimate is drifting *upward* away from
0.244 as `T` grows.

**What is and is not new.** That Rule 30's left side is visually regular is well
known and is in NKS. What this adds is: the closed diagonal recursion and its
period-propagation lemma (Observation 0), the settling law with a fitted and
independently replicated constant, the sharp period-cap-invariant boundary, the
clean binary entropy dichotomy against a matched random control, the O(1) pattern
map and the resulting `O(t)`-description of the wedge, and the observation that
the whole structure is disjoint from the prize object.

## Next Step

1. ~~Independent replication~~ **DONE** — `s = 1.3411` vs `1.3389`, all
   correctness gates PASS. See `experiments/wedge_verify.py` and
   `data/wedge_verify_T65536.json`.
2. **Push `T`** to 2·10⁵–10⁶ to pin the drift and decide whether `s` converges to
   a recognisable constant. Needs an incremental settle-time algorithm — the
   current `O(T·W)` packed buffer is memory-bound above `T ≈ 10⁵`.
3. **Direct `λ_L` comparison** as a designed experiment rather than an
   observation: measure the ensemble damage velocity and the single-seed
   structural boundary under one harness with matched estimators and error bars.
4. **Prove the period-16 conjecture** — but **not** by the finite-orbit route.
   The pair space is finite (`2^32`) so the orbit must cycle eventually, but it
   shows 11999/11999 distinct pairs with no sign of a repeat (Observation 6); a
   cycle could be `~2^32` steps away. The live route is the lemma's
   period-doubling branch: bound how often a diagonal can be eventually all-zero
   over a full period. That would close the conjecture to a Theorem and promote
   the wedge to a Certificate.
5. **Do not** extend this to "more randomness tests on the core." The core's
   incompressibility under three separate estimators is already recorded here;
   another aggregate test has no promotion path (`docs/theory/README.md` §5).

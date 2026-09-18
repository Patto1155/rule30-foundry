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

The active documentation uses this official numbering. Some archived handovers
retain the older swapped labels as historical records; do not copy those labels
into current work.

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

One-line form: searching a model class `M` against `n` bits and finding no fit
**discriminates this sequence only if `log2|M| >= n`**. Below that, a uniform
random string fits with probability at most `2^(log2|M| - n)`, so almost every
sequence gives the same answer and the search has measured `|M|`.

*Almost* every, not every: a 1-state DFAO generates the all-zero string at any
length. An exhaustive search that finds no fit has proved its exclusion, and
that proof does not weaken as the class shrinks — it simply is not evidence
about Rule 30. The two questions are separate and the gate now answers them
separately (`purpose: exact-exclusion`).

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
| Any bounded-period claim for the left diagonals | **Closed - the period is UNBOUNDED.** `period(d) ~ 2*log2(d)`; period 32 first appears at d=87867. | `2026-08-19-period16-refuted.md` |
| Finite-orbit / Floyd cycle certificate for period-16 | **Closed** — the state map is *partial*: the orbit leaves the deterministic region at every zero word (rate `2^-16`), so there is no cycle to find. `P[a trajectory survives long enough to cycle] ~ 10^-14231`. | `2026-08-19-orbit-cycle-structure.md` |
| Exhaustive `2^32` state-graph reachability sweep for period-16 | **Closed** — same flaw: assumes a total map on `(w_{d-2}, w_{d-1})`, which is not a complete state. | `2026-08-19-orbit-cycle-structure.md` |
| Low-degree GF(2) annihilator over center-column windows, `d <= 3`, `w <= 64` | **Closed** — full monomial rank at every parameter that clears both gates. Extends the LFSR row two degrees. | `2026-09-03-algebraic-annihilator.md` |
| Annihilator search at `w <= 22` (10M prefix) | **Vacuous by construction** — more distinct windows than the `2^w - 2^(w-d)` zeros a nonzero degree-`d` polynomial can have (Reed–Muller minimum distance). The negative is forced for *any* sequence. At `w <= 18` the windows cover `GF(2)^w` outright. | §5 below |
| Annihilator search over **space-time** patches | **Vacuous by construction — a forced positive.** The local rule is itself a degree-2 relation on a 2-row patch, so all 6 of its instances lie in the kernel on a 2x8 patch (0 violations) and the search succeeds whatever else is true. A search whose success is guaranteed by the definition of the object measures nothing. **Not** established: that the kernel is *only* the rule ideal — the kernel is 30-dimensional there and only 18 dimensions are attributed, the enumerated ideal slice being a lower bound. Deciding the rest is ideal membership, a Groebner problem, not a rank computation. `python experiments/algebraic_annihilator.py --space-time 8` | §5 below |
| **Centre-column automaticity ⇒ bounded diagonal period** (a contradiction proof for Prize 1/3) | **Closed — the bridge does not exist, and cannot be repaired at this hypothesis strength.** | Rejected 2026-09-18. The tempting chain is: assume the centre has a finite description; derive that diagonal behaviour must be bounded/regular; show Rule 30 diagonals violate it. **Two independent failures.** (1) *The pigeonhole step needs periodicity, not automaticity.* Assuming `(col_0, col_1)` eventually `p`-periodic, tail-pairs take at most `2^(2p)` values, so the tail map is periodic in `i`, the left half-plane tail is doubly periodic, and every diagonal has period dividing `pq` — bounded. Substitute "2-automatic" for "eventually periodic" and this evaporates: automatic sequences have no bound on their number of tails. What does survive is only a closure property — automatic sequences are closed under shift and pointwise Boolean combination, so `col_{-1}(t) = col_0(t+1) XOR (col_0(t) OR col_1(t))` is 2-automatic and by induction so is every `col_i, i<0` — but the state count multiplies at each step (`|col_{-1}| <~ |col_0|*|col_1|`), so there is no pigeonhole, no spatial periodicity, and no bounded period. (2) *Even granting the conclusion, it contradicts nothing.* Every left diagonal is **already proved** eventually periodic, and every eventually periodic sequence is `k`-automatic for every `k`. So "the diagonals are automatic" is already true independently of any assumption about the centre. A bridge whose conclusion is a known theorem can never yield a contradiction; unbounded periods across the family are fully compatible with every member being automatic. Automaticity is simply too weak to bound a period. The only hypothesis that would restore the bridge is **joint 2D automaticity of the space-time array** (which would force uniformly bounded complexity across the diagonal family, hence bounded period). That is strictly stronger than "the centre column is 2-automatic", it gives neither Prize 1 nor Prize 3, and it still sits behind two unproved inputs — unbounded `period(d)` (Robust observation, not Theorem) and a threshold lemma controlling the linear growth of `settle(d)`. Separately, the periodicity version proves only a statement about **two adjacent columns** and cannot reach Prize 1, which concerns `col_0` alone: Rule 30 is left-permutive but not right-permutive, so leftward reconstruction needs two columns while rightward reconstruction branches wherever `a(t,0)=1` — about half the time, giving `~2^(T/2)` consistent extensions over `T` steps, which is no constraint. | §3 above; `2026-08-19-period16-refuted.md` |
| **Any route to Prize 1 through automaticity or the 2-kernel** | **Closed — strictly harder than the equivalent form.** | Rejected 2026-09-18. Over a finite field a power series is rational **iff** its coefficient sequence is eventually periodic, so **Prize 1 is exactly the irrationality of `sum c_n x^n` over `F_2(x)`**. Eventually periodic implies rational implies algebraic implies 2-automatic, and each implication is strict. Proving non-2-automaticity therefore proves something strictly stronger than Prize 1 needs. Work the rationality form (`D=1`); automaticity and the 2-kernel bear on Prize 3 only. | §2, and the certified `L(n)=n/2` row in `CLAIM_LEDGER.md` |
| "Every left diagonal is eventually periodic" as a shortcut | **Closed** — settling time grows linearly, centre column never settles | §3 above |
| **Right-special factors** — "prove that for every `n` the centre column has a right-special factor of length `n`" | **Not a route — it is Prize 1 restated, at equal strength.** Admissible as a reformulation, worthless as a reduction. | Audited 2026-09-18. Write `p(n)` for the number of distinct length-`n` factors of the centre column `c`. Every occurring factor of an infinite word has at least one right extension, so `p(n+1) = sum over w in L_n of #ext(w)`, giving `p(n+1) - p(n) = #{right-special factors of length n}` exactly (binary alphabet). **Morse-Hedlund**, *Symbolic Dynamics*, Amer. J. Math. **60** (1938) 815-866, **Thm 7.4** (as cited and restated in Kopra 2023, Thm 4.1): if `x` is not eventually periodic then `p(n) >= n+1` for every `n`. Contrapositive plus stabilisation — `p(n+1) = p(n)` means every length-`n` factor has exactly one right extension, hence so does every length-`(n+1)` factor, so `p` is constant from `n` on and therefore bounded — gives: eventually periodic iff `p` bounded iff `p(n+1) = p(n)` for some `n`. Therefore **"right-special factor at every length" is literally equivalent to "not eventually periodic"** — it is Prize 1 with no loss and no gain, a third equivalent form beside the `F_2(x)` irrationality row above. Unlike the automaticity row it is not *strictly harder*, so it is not closed; it is simply not easier. Three structural facts to carry into any attempt. **(a) The induction direction is backwards.** Right-specialness is suffix-closed (`w0, w1` factors implies `u0, u1` factors for every suffix `u` of `w`), so `RS(n+1) => RS(n)`; the set of valid `n` is downward closed and failure is a single threshold. An induction needs `RS(n) => RS(n+1)`, which is false for words in general — quickest counterexample `c = 1 0^inf`, which has `RS(0)` and fails `RS(1)`. Equivalently (Koenig), `RS(n)` for all `n` iff the suffix tree of right-special factors is infinite iff there is a **left-infinite word all of whose finite suffixes are right-special**; that branch, not a per-`n` step, is what a construction must produce. **(b) Any argument from the recurrence alone proves too much.** The all-zero configuration is a fixed point (`0 XOR (0 OR 0) = 0`), centre column `0^inf`, `p(n) = 1`, `RS(0)` already false; every finite cyclic-ring orbit likewise has an eventually periodic centre column. So a proof invoking only the ANF, left-permutivity, or the diagonal recursion — anything invariant under change of initial condition — is refuted in one line. Sharper still, and this is the decisive one: **Rule 90 is left permutive too, and from the very same single-cell seed its centre column is eventually periodic** (Kopra 2023, §4, closing remark; the centre bit is `C(2k,k) mod 2`, which is 0 for every `k >= 1`). So permutivity cannot separate the two rules on the prize configuration itself. The separating ingredient is the **OR latch** in `a(t,i) OR a(t,i+1)`, which an additive rule does not have (Condrey, Fig. 1 and §3). This is the same failure the strip rows in `CLAIM_LEDGER.md` record: constructions valid with free boundary, rejected by the actual seed. **(c) The permutivity engine is already spent, and it stops at pairs.** The step the recurrence suggests — `c_{t+1} = a(t,-1) XOR (c_t OR a(t,1))` with `a(t,-1)` entering by pure XOR, so two occurrences of a window that agree on `(col_0, col_1)` but differ in the left driver diverge — is, run to completion, the sideways-determinism argument, and it yields **Jen (1990), Prop. 3: for Rule 30 and any number-like configuration, the width-2 trace — any two adjacent columns — is not eventually periodic.** Verified against Kopra, *TCS* **946** (2023) 113668, Thm 3.5, which restates and reproves it (Rule 30 is `(1,1)` left permutive, hence left expansive with dimensions `(0,1,2)`, so `w = 2`), and which states outright that **width 1 is open** (Problem 4.8). So this engine reaches width 2 and stops one short of the prize, which is width 1. Rule 30 is not right-permutive, so `col_0` alone leaves `~2^(T/2)` consistent `col_1`, and the pair statement does not descend to the single column. **The gap is real, not technical:** Rule 90 is also rapidly left expansive, so its width-2 traces are aperiodic by the same theorem — yet from the *identical* single-cell seed its centre column is eventually periodic (Kopra, §4, closing remark). Same seed, same permutivity, opposite answer at width 1. **Ensemble trap, new costume — and it is provably fatal here.** The factor-complexity literature works the *trace subshift* (all initial configurations), and for Rule 30 that object is the **full shift**: Condrey (arXiv:2609.09431, 2026) §5 observes — as a remark with a one-line proof, not a numbered theorem, though the one line is just left permutivity inverting cell by cell leftwards — that for any prescribed trace of length `N` and any compatible right prefix there is a unique left half, hence a finite row of support radius `~N` realising that trace through time `N`. So every binary word is a trace, the trace subshift has complexity `2^n`, and every factor of every length is right-special **in the ensemble** while saying nothing whatever about the one seed. This is §0's rule with a theorem behind it rather than a policy. Do not substitute it. **Where the real ladder is.** Excluding eventual period `p` is a genuine partial result and the literature climbs that ladder. `p = 1` is **settled for every nonzero finite configuration** — no column of such an orbit is eventually constant (Condrey 2026, Cor. 5) — which gives `RS(0)` for the centre column as a theorem rather than a prefix measurement. **Split the credit correctly:** the eventually-*one* half follows from Jen (1986), Thm 7a, and Condrey disclaims novelty for it explicitly; what is new there is the eventually-*zero* exclusion, the two constant-trace fibers, and the sharp horizons. **And note the status:** arXiv:2609.09431 is a preprint of 8 Sep 2026, not refereed, whose own appendix states that its Lean file is "supporting verification, not a machine-checked proof of the main theorems". Treat `RS(0)` as resting on that. `p = 2` is open — Condrey, §5: "the next unresolved case is eventual period two; we make no claim about it or any higher period". Kari is thanked there for suggesting the exclusion of **nonconstant periodic traces** as the natural next step, which is every `p >= 2`, not `p = 2` alone. The `n`-ladder is not a ladder: `RS(n)` is downward closed, so it collapses at a single threshold, and `RS(N)` failing means an eventual period of at most `p(N) <= 2^N`. Reaching `RS(N)` through periods therefore costs exponentially more than it returns — but that direction has proofs and this one has none. **What is free, and its ceiling.** The certified window counts in the annihilator log give `p(n) = 2^n` for `n <= 18` and `p(20) >= 1,048,480 > 2^19`, `p(22) >= 3,806,484 > 2^21`, which force `p(n+1) > p(n)` and hence `RS(n)` **for every `n <= 21`**, with no new compute. No finite prefix can do much better: a witness for `RS(n)` is a repeated length-`n` factor with divergent continuations, so a length-`N` prefix can exhibit `RS(n)` only for `n` up to its longest repeated factor, `~2*log2(N)`. That ceiling is **certified, not estimated**: the same table records `p(64) >= 9,999,937` on the 10M prefix, which is every one of its `10^7 - 63` length-64 windows, so the prefix contains **no repeated 64-block** and can witness `RS(n)` only for `n <= 63`. A sweep therefore cannot reach large `n`, and failure to find a witness is never a counterexample (witnesses may lie deeper). **Finite sweeps have no falsification power against this lemma** and must not be run against it. | Morse-Hedlund, *Amer. J. Math.* **60** (1938) 815-866, Thm 7.4; Jen, *Physica D* **45** (1990) 3-18, Prop. 3 (read through Kopra Thm 3.5; Jen not consulted directly) and *J. Stat. Phys.* **43** (1986) 219-242, Thm 7a (via Condrey, not consulted directly); Kopra, *Theor. Comp. Sci.* **946** (2023) 113668, Thm 3.5 and Problem 4.8; Condrey, arXiv:2609.09431 (2026, preprint), Cor. 5 and §5; `docs/experiment-logs/2026-09-03-algebraic-annihilator.md` window table; §0, §3 above |

---

## 5. The annihilator gate (do not re-derive this)

An annihilator of a bit stream is a nonzero polynomial `f` of degree `<= d` over
a `w`-bit sliding window with `f(s[i..i+w-1]) = 0` at every position. Searching
for one is a rank computation, and it can be vacuous in **both** directions.

**Forced positive.** The monomial basis has `D = sum_{k<=d} C(w,k)` elements.
Fewer than `D` independent rows leaves a kernel by dimension alone. This is the
Admission Rule with `log2|M| = D` against the number of distinct windows.

**Forced negative.** The minimum distance of the Reed-Muller code `RM(d, w)` is
`2^(w-d)`, so a nonzero polynomial of degree `<= d` takes the value 1 at least
`2^(w-d)` times and therefore vanishes **at most `2^w - 2^(w-d)` times**. An
annihilator must vanish on every observed window, so if the stream shows more
distinct windows than that, no annihilator can exist *whatever the sequence is*.

This is a theorem, not an estimate, and on the 10M prefix it is the binding one:
it voids every search at `w <= 22`. Both gates live in
`experiments/counting_bound.py` (`annihilator_verdict`, `max_zeros_of_degree`).

**Width is the cheap axis.** The ceiling grows as `2^w` while `D` grows
polynomially, so widening a search is gated far more loosely than deepening it:
`w=64, d=2` needs `D=2081` columns and runs in seconds, while `w=32, d=4` needs
`D=41449`. Widen before you deepen.

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

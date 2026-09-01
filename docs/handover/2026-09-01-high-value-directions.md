# Highest-value research directions — 2026-09-01

This document is the current priority order after the first Nersissian end-to-end audit.

The central resource-allocation rule is:

> Prefer experiments that can expose a shortcut, contradiction, or reusable lemma over experiments that merely extend a negative empirical horizon.

Raw compute is not currently the bottleneck. The bottleneck is having a hypothesis worth scaling. Do **not** rent compute until a direction below produces a reason to do so.

## Priority summary

| Rank | Direction | Suggested effort | Prize-facing value |
| --- | --- | ---: | --- |
| 1 | Faithful Nersissian compressed-representation audit | 30% | Directly probes Problem 3 shortcut complexity |
| 2 | Problem 1 periodicity -> contradiction search | 25% | Has a plausible computation -> lemma -> theorem path |
| 3 | Corrected circuit/index-function programme | 20% | Probes finite complexity of the map `i -> c_i` |
| 4 | Cross-rule controls | 15% | Determines which existing complexity signals are Rule-30-specific |
| 5 | DFAO / grammar / GF(2) diagnostics | 7% | Cheap structural reconnaissance; mostly model-class negatives |
| 6 | Larger simulations / discrepancy | 3% | Useful only when tied to a specific falsifiable hypothesis |

Engineering prerequisites for future paid runs are listed separately below. They are mandatory gates, not research directions.

---

## 1. Faithfully reconstruct and audit Nersissian's compressed representation

### Why this is first

The current branch implements the explicit support recurrence and establishes the accounting framework, but **does not yet reconstruct the masked/dyadic compressed representation behind the advertised fast lookup**.

That is now the most prize-facing unresolved question in the repository:

> Starting with only `n`, can the complete machinery required to return center bit `c_n` be constructed and queried in `o(n)` work without hiding `Omega(n)` n-dependent information in preprocessing or storage?

A genuine sublinear cold-start implementation would be major progress on Problem 3. A precise demonstration that the fast query requires linear-or-worse construction would also be valuable because it identifies exactly where the apparent shortcut stops being a shortcut.

### Required implementation

Reconstruct the primary-source representation faithfully enough that every operation can be mapped back to a documented mathematical step. Put it behind the same accounting interface as `experiments/nersissian_audit.py`.

Measure separately:

- n-dependent representation construction work;
- representation size and peak memory;
- recursion depth and intermediate-object sizes;
- warm query work after the representation exists;
- cold isolated `n -> c_n` work;
- sequential/amortised work across nearby queries;
- whether any state reused between queries itself required `Omega(n)` prior work.

Prefer deterministic operation counts to timings whenever possible.

### Correctness requirements

For every tested `n`, compare against the repo's independent Rule 30 center engine. Do not use precomputed center-column values inside the candidate algorithm itself.

The experiment must keep these cases separate:

1. fresh state -> isolated `c_n`;
2. prebuilt representation -> `c_n`;
3. sequential `c_0, ..., c_n`;
4. `c_(n+1)` after already computing `c_n`.

### Decision gate

**Escalate immediately** if a faithful implementation shows credible `o(n)` cold-start work *and* sublinear n-dependent storage. Increase validation horizons, derive the recurrence carefully, seek independent reimplementation, and treat it as a potential Problem 3 shortcut.

**If construction is linear or worse**, identify the specific recurrence/object causing that cost and try to turn the observation into a structural complexity argument. Do not claim an asymptotic lower bound from timing regression.

**If the primary source is ambiguous**, stop before inventing missing mathematics. Record the ambiguity precisely and, if possible, construct multiple interpretations as explicitly labelled variants.

---

## 2. Problem 1: search for periodicity contradictions rather than longer period-free prefixes

### Objective

Move from

> "no period found through N bits"

into a theorem-shaped implication:

> eventual center period `p` -> forced Rule 30 spacetime constraints -> contradiction.

Wolfram-scale center-column computation already makes another large finite negative comparatively low-value. The useful computational target is a **repeated obstruction that can be generalized in `p`**, not a larger horizon.

### Proposed programme

Build SAT/SMT instances that impose:

- exact Rule 30 local constraints;
- an eventually periodic center column with period `p`;
- controlled transient length;
- enough surrounding spacetime width to force consequences.

Sweep small-to-moderate values of:

- period `p`;
- transient length;
- surrounding width/depth;
- phase where useful.

For UNSAT instances:

- extract/minimize proof cores;
- classify recurring local obstruction motifs;
- test whether the same obstruction persists as `p` grows;
- attempt to state the obstruction symbolically rather than numerically.

### Deliverable that matters

The target output is not an UNSAT table. It is a candidate lemma of the form:

> any period-`p` center tail forces configuration/property X, but Rule 30 local constraints forbid X.

A machine-checkable family of proof cores that suggests such a lemma is substantially more valuable than another period search.

### Stop condition

If proof cores change chaotically with `p` and no reusable structure appears after a deliberate bounded sweep, downgrade the direction rather than simply increasing SAT size.

---

## 3. Correct the circuit direction: study the entire index function

### Do not use the broken fixed-n definition

For fixed `n`, the "smallest circuit computing center bit `n`" is trivial: hard-code the answer bit.

The meaningful finite object is the function mapping a `k`-bit index to the corresponding center bit:

```text
C(k) = min circuit size such that
       C(i) = c_i for every 0 <= i < 2^k.
```

The circuit receives the `k` binary digits of `i`.

### Why this matters

This asks how complicated the entire finite index-to-center-bit function is. It is much closer to shortcut discovery than compressing one fixed prefix and naturally captures XOR/parity structure that several existing model classes miss.

It is still **non-uniform** complexity. A large finite circuit lower bound does not by itself prove a uniform `Omega(n)`-time lower bound for Problem 3.

### Concrete tasks

1. Define an explicit gate basis and size metric.
2. Include parity-capable gates or compare bases with/without XOR so the experiment is not blind to GF(2) structure.
3. Exactly synthesize/minimize circuits for the largest tractable `k` using SAT/SMT/e-graphs or another checkable method.
4. Verify every synthesized circuit on all `2^k` inputs.
5. Compare Rule 30 against simple, automatic, linear, and chaotic-rule controls.
6. Track `C(k)` and structural motifs in optimal/near-optimal circuits.

### Escalation condition

Escalate if Rule 30 exhibits a reproducible circuit-growth or structural signature that distinguishes it from appropriate controls, or if synthesis discovers a reusable shortcut pattern.

If all chaotic controls show the same curve, use that result to stop spending substantial time on the model class.

---

## 4. Cross-rule controls as a research-direction filter

### Question

The repository has accumulated many results of the form:

> Rule 30 resists model class X.

Without chaotic-rule controls, that cannot distinguish:

- something special about Rule 30; from
- a limitation of model class X on chaotic ECAs generally.

### Minimum control panel

Run the same certified machinery over at least:

- Rule 30;
- Rules 45, 73, 89, 105, 110;
- appropriate simple/linear controls;
- random-sequence controls where mathematically meaningful.

Prioritize comparison of:

- minimum DFAO state curves `s*(n)`;
- LFSR / GF(2) complexity;
- grammar complexity;
- corrected circuit/index complexity;
- parity-capable predictors/estimators.

### Decision rule

If Rule 30 is a clear outlier under a measure, chase that measure mechanistically.

If all chaotic rules are indistinguishable, treat the measure as a **direction filter** and stop scaling it solely on Rule 30.

This work is valuable because it can prevent months of scaling an uninformative model class.

---

## 5. Cheap DFAO / grammar / GF(2) diagnostics

These remain useful because they are inexpensive and produce checkable finite artifacts, but they should not dominate the programme.

### Worth doing

- extend `s*(n)` modestly beyond the current certified horizon;
- close the base-2-only gap with bases 3/4/5 where feasible;
- continue exact smallest-grammar work at small `n`;
- add parity/GF(2)-capable estimators where existing neural/model suites are structurally blind;
- preserve SAT/DRAT or equivalent certificates wherever a finite negative is claimed.

### Interpretation rule

These are finite model-class exclusions, not proofs of computational irreducibility. Do not promote "random-like finite complexity" into a prize claim.

---

## 6. Larger simulations and discrepancy: only with a sharp hypothesis

Large center-column runs are now lowest priority unless they test a specific prediction that cannot be tested with existing data.

### Still worth doing

- finish the cheap left-edge period-doubling walk because it tests a sharp prediction and costs little;
- compute finite discrepancy certificates when they close a documentation gap;
- extend a simulation horizon only when a preceding theory/model predicts a concrete event, transition, or scaling break at that horizon.

### Not worth doing

- renting a larger GPU simply to report "no period through a bigger N";
- increasing horizons because VRAM is available;
- using larger empirical negatives as substitutes for asymptotic arguments.

---

# Engineering gates before any paid compute

These are mandatory before renting hardware, regardless of research direction.

## A. GPU simulator checkpoint/resume

The GPU simulator still lacks tested checkpoint/resume. A preempted paid run can lose all work. Implement and validate resume semantics before spot/preemptible compute.

## B. Eliminate the current `verify_all` SKIP on the rental machine

`SKIP` is not `PASS`. Determine exactly which verification stage is currently skipped and ensure the intended rental environment reaches a no-SKIP preflight state for all artifacts relevant to the run.

## C. Re-run full verification before and after changes

Required command:

```bash
python tools/verify_all.py
```

The GitHub connector used to prepare this branch cannot execute that command, so a shell-capable agent or local checkout must run it before merge.

## D. Keep tape-geometry validation mandatory

Any future GPU run must satisfy the light-cone bound before launch. Silent late corruption from an undersized tape is already a known failure mode.

---

# Recommended next-session order

A fresh agent should execute work in this order unless new evidence changes the ranking:

1. **Reconstruct Nersissian's compressed/masked-dyadic representation faithfully.**
2. Validate it against the independent Rule 30 engine and the published examples.
3. Instrument cold, warm, sequential, memory, and deterministic operation counts.
4. Run only enough scaling points to identify what recurrence needs mathematical analysis.
5. In parallel or immediately after, build a small Problem 1 periodic-center SAT prototype designed around proof-core extraction.
6. Formalize the corrected circuit/index-function definition and prototype exact synthesis at small `k`.
7. Run cross-rule controls before investing heavily in any newly observed complexity curve.
8. Complete cheap diagnostics only when they do not displace the top four directions.
9. Spend **$0 on rented compute** until one of the above produces a specific experiment that local hardware cannot answer.

# What would change this ranking?

Promote a direction if it produces any of:

- a credible sublinear center-bit algorithm;
- a reusable symbolic recurrence suggesting such an algorithm;
- a repeatable UNSAT obstruction that plausibly generalizes to arbitrary period;
- a Rule-30-specific complexity signature absent from chaotic controls;
- a proof-candidate lemma with machine-checkable finite evidence.

Downgrade a direction if repeated work only yields:

- a larger negative prefix;
- another generic "looks random" statistic;
- the same behavior across chaotic ECA controls;
- finite model-class exclusions with no new structural mechanism.

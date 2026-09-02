# Compute Plan — What to Rent, and What Not To

**Date:** 2026-08-30
**Model:** `python tools/plan_run.py --table` (calibrated on this repo's own runs)
**Read with:** `docs/CLAIM_LEDGER.md`, `docs/theory/finite-prefix-counting-bound.md`

This is a decision document. It exists because "rent a bigger GPU" is the most
expensive way this project can be wrong, and because the standing plan's
premise for renting does not survive contact with the measured numbers.

> **Companion document.** `docs/handover/2026-09-01-high-value-directions.md`
> (PR #20) ranks the *research* directions and is the authority on what to
> work on. This file is the authority on what hardware to buy and what a run
> costs. Where they overlap they agree, with one exception: that document
> corrects D11 below, and the correction is applied here. Read both.

---

## 1. The premise for renting was wrong

`docs/handover/2026-08-30-next-session-plan.md`, item 13:

> The binding constraint is VRAM: the light cone spreads at speed 1, so `N`
> valid center bits needs ~`2N` cells. […] a 24 GB card quadruples the
> reachable horizon over the local 6 GB GTX 1060.

The repo's own artifacts say otherwise:

```
data/center_col_46M_results.json:  93,000,000 cells x 46,000,000 steps
                                   7,106 s,  602 Gcells/s,  VRAM 66.0 MB
```

**66 MB of a 6 GB card — 1.1%.** Simulation cost is *quadratic* in the horizon
while memory is *linear*, so time exhausts long before memory does:

```
         steps          cells      VRAM   GTX 1060      RTX 4090     A100 40GB      H100 SXM
    10,000,000     20,000,064      14 MB      6 min         1 min         1 min         0 min
    46,000,000     92,000,064      66 MB      2.0 h        23 min        15 min         7 min
   100,000,000    200,000,064     143 MB      9.4 h         1.8 h         1.2 h        32 min
   300,000,000    600,000,064     429 MB      3.5 d        16.1 h        10.5 h         4.9 h
 1,000,000,000  2,000,000,064    1431 MB     39.2 d         7.5 d         4.8 d         2.2 d
```

A 10⁹-step run still fits the existing 6 GB card and would take 39 days on it.
**VRAM never binds anywhere reachable. Renting for VRAM buys nothing.**

### What to select hardware on

At 590 Gcells/s over 64 cells per `uint64`, the kernel does ~9.2e9 word-updates
per second, each reading three words and writing one: ~295 GB/s of traffic
against the GTX 1060's 192 GB/s peak. It is at the bandwidth roof.

**Rent on memory bandwidth. Ignore FLOPS, ignore VRAM.** `tools/plan_run.py`
scales linearly from the measured baseline, which reproduces both recorded runs
to within 3%.

One consequence worth stating: the **RTX 2050 laptop is slower than the GTX
1060** for this workload (112 vs 192 GB/s). Its value was cross-hardware
verification, not throughput.

---

## 2. Most of the best work costs nothing

Ranked by insight per dollar, the top of the list is free. **Do these before
renting anything** — several of them change what is worth renting for.

| # | Direction | Cost | Why it is worth doing |
|---|---|---|---|
| **D1** | **Finish the item-14 pattern-map walk to `d = 1.2e10`** | **~26 min CPU per branch** | Tests a *sharp falsifiable prediction* — `period(d) ~ 2·log2(d)` says a 32→64 doubling near `d ≈ 8.6e9`. The tool exists, gates pass, validated to `d=5e7`. Cheapest conclusive result available anywhere in the repo. Caveat: the ledger grades left-edge structure as **disjoint from the prize object**, so this is decisive mathematics that cannot by itself yield a center-column shortcut. |
| **D2** | Extend `s*(n)` past n=48 (item 10) | hours, CPU | Extends a **Certificate**. The SAT work measured ~28× cheaper than the plan assumed (207 solves in 133 s vs 3747 s for 105 through pysat). **Grow the state budget with `n`** — raising `n` at fixed `--max-states` makes the negative *more* vacuous. |
| **D3** | `s*(n)` in bases 3, 4, 5 (item 9) | hours, CPU | The new Certificate covers **base 2 only**, and automaticity is base-dependent (Cobham). This closes a real scope gap in a claim already at the top grade. |
| **D4** | Problem 3 as a finite discrepancy bound (item 12) | minutes, existing data | Converts the repo's weakest headline ("behaves close to fair") from Observation to Certificate, using the 10M bits already on disk. Compute `D_N = max_k |S_k − k/2|` against `sqrt(N log log N)`. Pure win, no new data. |
| **D5** | Exact smallest grammar at small `n` | hours, CPU/SAT | `g(n)` is an *upper-bound* curve; the counting null and Re-Pair bracket the truth ~2.5× apart. Exact solving at n ≤ 64 promotes it to Certificate by the same route `s*(n)` took. |
| **D6** | Parity-capable estimators on the center column | hours, CPU | The direct answer to item 8's blind spot: the neural suite cannot see `s[i-13] ⊕ s[i-27]` at any budget tested. Berlekamp–Massey and GF(2) rank find exactly that. **A model-class blind spot is answered by a different method, not a bigger network.** |
| **D7** | 46M independent golden reference | ~50 h on a cheap CPU box (~$5) | Now possible: `gen_golden_reference.py` checkpoints and resumes as of this session. Closes the last verification gap — the 46M artifact's independent check currently reaches only its first 10M bits. |

---

## 3. What renting actually buys

Only three directions genuinely need paid hardware, and they are not equally
good.

### D8 — A longer center column (Problem 1, and every prefix curve)

`--steps 3e8` is **~10.5 h on an A100 (~$15–20)** or **4.9 h on an H100**.
That single artifact feeds:

- the exact period search (already exhaustive and exact; the scan itself takes
  **2.1 s** — it is the *column* that is expensive, never the search),
- every prefix-based curve at larger `n`,
- the block/entropy estimators.

**Be honest about what it is.** A period search over a longer prefix is buying
a *bigger negative*. Eventual periodicity is asymptotic; no finite prefix can
settle Problem 1. The expected outcome is "no period", and that outcome is
worth having only because it is cheap and because the alternative outcome is a
$30,000 counterexample. Budget it as a by-product of generating the column, not
as the reason for it.

### D9 — Cross-rule comparison: do our measures discriminate at all?

**The strongest new direction, and it is embarrassingly parallel.**

Every negative in this repo is about Rule 30 alone: `s*(n)`, `g(n)`, `L(n)`,
A–L. None of them has ever been run against another chaotic ECA. So the repo
cannot currently distinguish

- "Rule 30 specifically resists these model classes" from
- "these model classes cannot compress *any* chaotic ECA, and Rule 30 is not
  special".

Run the certified curves over rules 45, 73, 89, 105, 110 (and a Bernoulli
control). If Rule 30 is an outlier on any measure, that is a lead worth
chasing. If every chaotic rule looks identical, the existing negatives are
correctly interpreted as statements about the *measures*, which is itself a
finding and would redirect the whole programme.

Costs almost nothing per rule and parallelises perfectly across cores — the
best fit for a rented many-core box rather than a GPU.

### D10 — The ML ladder at original scale

Settles whether item 8's blind spot is architecture-limited (as six budgets
here suggest) or merely budget-limited. **~$5–15 on one GPU-hour.** Cheap, and
it either hardens a caveat that currently qualifies three README rows, or
weakens it. Either way the README changes.

---

## 4. A genuinely new direction worth designing before buying

### D11 — Circuit-size curve `C(k)`

> **Corrected 2026-09-02.** The first version of this section defined
> `c(n)` as "the smallest circuit computing center bit `n`". That is
> **vacuous**: for a *fixed* `n` the answer is a hard-coded constant, size
> O(1), for any sequence whatsoever — the same failure mode as the retracted
> 1–4 state DFAO certificate. Caught in
> `docs/handover/2026-09-01-high-value-directions.md` §3. The definition below
> replaces it.

The repo has measured minimal-description curves over three model classes:
LFSRs (`L(n) = n/2`), DFAOs (`s*(n)`, Certificate), and straight-line grammars
(`g(n)`). The natural fourth is the one closest to Problem 2's actual wording —
but it has to quantify over the whole index function, not one index:

    C(k) = size of the smallest boolean circuit that, given the k binary
           digits of i, outputs c_i correctly for EVERY 0 <= i < 2^k

This matters because Rule 30's update **is** a circuit — its ANF is
`a(t,i-1) ⊕ a(t,i) ⊕ a(t,i+1) ⊕ a(t,i)·a(t,i+1)`, two gates. "A faster
algorithm for the nth cell" is, formally, a small circuit family. It also
connects to D6: this is the model class that *contains* the XOR structure the
neural suite is blind to, so it probes precisely the gap item 8 exposed.

**Admission Rule, before running.** With 2-input gates, gate `j` chooses an
ordered pair from the `k` inputs plus the `j-1` earlier gates and one of 16
types, so `|M(s)| <= (16 (k+s)^2)^s` and
`log2|M(s)| ≈ s(4 + 2 log2(k+s))`. The function being fitted is a truth table
of `2^k` bits, so a negative is informative only once that exceeds `2^k`:

| k | bits to fit | counting threshold | Shannon expectation `~2^k/k` |
|---|---|---|---|
| 5 | 32 | s ≈ 4–5 | ≈ 6.4 gates |
| 6 | 64 | s ≈ 6 | ≈ 10.7 gates |

Measured minima should sit *above* the threshold, as they do for `s*(n)`.

**The honest limitation, stated before anyone spends on it.** Exact circuit
synthesis is bounded by the truth table, not by a prefix: `k = 6` already means
minimising over a 64-bit function, and the cost grows doubly. So the reachable
curve is roughly `C(1..6)` — which covers only the **first 64 center bits**,
*less reach than the 128-bit DFAO claim that was retracted for vacuity*. It is
not vacuous here, because the counting bound is satisfied, but a six-point
curve is thin evidence next to `s*(n)` to n=48 or `g(n)` to n=65536.

It is also **non-uniform** complexity: a finite circuit lower bound does not
imply a uniform `Ω(n)`-time lower bound for Problem 3.

Prototype it for free at `k <= 5` and let the result decide whether it deserves
more. Do not budget it as a headline direction before that.

---

## 5. Pre-flight checklist before spending anything

Three bugs fixed this session were all in the path a long paid run takes.
Re-check them on the rented box before starting:

1. **`python tools/verify_all.py`** — must be all PASS with no SKIP on a
   machine holding the bitstreams. `SKIP` is not `PASS`.
2. **Light-cone sizing.** `gpu/rule30_sim.py` now refuses a tape too short for
   its step count. A short tape does **not** crash, keeps a 0.5 bit mean, and
   passes the first-20-bit check — it fails *late*. Verify with
   `python gpu/tape_geometry.py --steps <N>` before launching.
3. **Checkpointing.** `gen_golden_reference.py` has `--checkpoint` and
   `--stop-after`; resumption is bit-exact and tested. The GPU simulator does
   **not** checkpoint yet — a preempted spot instance loses the whole run. Fix
   that before buying spot time for anything over an hour.
4. **Output path.** `--center-out` with a bare filename used to crash *after*
   the simulation finished. Fixed, but pass a path with a directory anyway.

### Suggested order

1. D1–D6 on the machines you already have. Several will change section 3.
2. D11 prototyped small and free; decide whether it scales.
3. Then one rental: a bandwidth-heavy GPU-hour for D8 + D10, and a many-core
   CPU box for D9 + D7.

Nothing in section 3 costs more than about $20. If a plan starts quoting
hundreds, re-read section 1 — it has almost certainly reintroduced the VRAM
assumption.

# Experiment Log — Why the Cycle-Certificate Route to Period-16 Is Vacuous

- Date: 2026-08-19
- Title: The pattern-map orbit leaves the deterministic region before it can cycle
- Claim Level: **Robust observation** (survey) + **Theorem** (the partial-map argument)
- Prize: Problem 3 context. Structural work on the seed orbit.

## Goal

Lemma C (`2026-08-19-settled-word-genericity.md`) established that the state map
`F(u,v) = (v,w)` is injective wherever `v ≠ 0`. That suggests an attractive
shortcut to deciding period-16:

> The orbit lives on a finite state space and `F` is injective, so the orbit is
> eventually periodic. Run Floyd cycle detection (`O(1)` memory, no
> precomputation) to find its cycle. If the cycle contains no odd-parity
> collision, period-16 is proved **forever** — a finite certificate, far cheaper
> than the `10^6` walk or an exhaustive `2^32` reachability bitmap.

The argument is valid *if the orbit is a deterministic orbit*. This experiment
tests that premise. It fails.

## The obstruction

`F` is a **partial** map. It is undefined at the `2^16` states with `v = 0`,
because there the one-period composite is affine and **two** period-16 words
satisfy the recursion. This is not a modelling nicety — it is verified in
`experiments/zero_word_regression.py` gate 3: at `d = 399` both `0x9f60` and
`0x609f` are consistent (they are exact bitwise complements), and only
simulation picks `0x9f60`.

> **`(w_{d-2}, w_{d-1})` is not a complete state.** Information from outside the
> 32 bits enters at every zero word. The completion is the diagonal *transient*,
> whose length grows like `settle(d) ≈ 1.34·d` — unbounded, so there is no
> finite state space that makes this deterministic.

Floyd requires a total function. So the question is not "how long is the cycle"
but "does the orbit ever reach a cycle before it exits".

## Setup

`experiments/orbit_cycle_structure.py`. A vectorized `f_vec` (two-period
convergence, valid because `v ≠ 0` guarantees a reset position) is gated
bit-exact against the trusted scalar `pattern_map_step`: **20000 pairs, 0
mismatches**. Then:

1. Floyd on the **real** seed orbit, started at the first state past the last
   early zero word, halting where `F` is undefined.
2. A lockstep survey of 4096 random start states, cap `2^17`, measuring
   trajectory length before termination against the predicted geometric law
   `P[survive n] = (1 - 2^-16)^n`.

## Result 1 — the real orbit exits; it does not cycle

```
Floyd on the seed orbit (start d=403)
  outcome : left_deterministic_region
  steps   : 52805
  reason  : v_is_zero
```

52805 steps from `d = 403` lands at `d = 53208`, consistent with the known zero
word at `d = 53207` produced by the first real collision at `d = 53205/53206`.
**No cycle. The orbit exits.**

## Result 2 — that is the generic behaviour, not bad luck

`F` is a partial injection with `2^16` terminal states out of `2^32`, so a
trajectory should terminate at rate `2^-16` per step. Measured:

| steps | survivors | geometric prediction |
|---|---|---|
| 256 | 4072 | 4080.0 |
| 1024 | 4016 | 4032.5 |
| 4096 | 3847 | 3847.8 |
| 16384 | 3194 | 3190.0 |
| 32768 | 2542 | 2484.3 |
| 65536 | 1443 | 1506.8 |
| 131072 | 521 | 554.3 |

The geometric law is followed across four orders of magnitude.

*Censoring note (per `AGENTS.md`):* the reported mean terminated length of
44894 is **right-censored** — 521 of 4096 trajectories were still alive at the
cap and are excluded from that mean, which biases it below the true `2^16`. The
survival curve above is the uncensored statistic and is the one to read.

## Result 3 — the cycle route is not merely expensive, it is vacuous

For a cycle to close, a trajectory must dodge **every** terminal state for about
the expected cycle length of a random permutation, `2^32 / 2 ≈ 2.15×10^9` steps:

```
P[trajectory survives that long]  ~  (1 - 2^-16)^(2.15e9)  ~  10^-14231
```

> The orbit provably keeps leaving the region where the cycle argument applies —
> roughly once every `2^16` diagonals, forever. There is no cycle to find.

This also **retracts the exhaustive-bitmap proposal** floated in
`2026-08-19-settled-word-genericity.md` ("a `2^32`-bit visited bitmap is 537 MB,
feasible on this machine"). It fails for exactly the same reason: it assumes a
total function on the 32-bit state. Both routes are closed by one observation.

This is consistent with, and explains, the pre-existing §4 entry in
`docs/theory/README.md` — *"finite-orbit proof: pair orbit shows 11999/11999
distinct, no cycle within reach."* An earlier session observed the symptom at
small scale; the mechanism is that terminal states are hit at rate `2^-16`, so
"within reach" was never the binding issue.

## Interpretation

Three routes to deciding period-16 have now been considered and two are closed:

| Route | Status |
|---|---|
| Floyd cycle certificate | **Closed** — orbit exits the deterministic region, `P[cycle] ~ 10^-14231` |
| Exhaustive `2^32` reachability bitmap | **Closed** — same flaw, assumes a total map |
| Walk to `d ~ 10^6`, resolving each branch point | **Open, and now the only survivor** |

The walk is well-founded: the `2^-16` rate is measured, and ~15 branch points
are expected below `10^6`.

### A cost note for the walk

Resolving a branch point does **not** require simulating the full 2D cone. The
diagonal recursion is exact for the actual sequences, not just their eventual
patterns:

```
D_d(t+1) = D_{d-2}(t) XOR (D_{d-1}(t) OR D_d(t))
```

so diagonals can be propagated with a **rolling window of three**, giving `O(T)`
memory instead of `O(d·T)`. Better, the recurrence self-resets: wherever
`D_{d-1}(t) = 1`, the update collapses to `D_d(t+1) = NOT D_{d-2}(t)`
independently of `D_d(t)`. Since diagonals are roughly half ones, dependency
chains between resets are short (~2 steps), so the scan vectorizes rather than
running as a serial `T`-step chain. This should be measured before the walk is
scaled, not assumed.

## Next Step

1. Scope the rolling-window diagonal propagator and verify it bit-exact against
   `left_diagonals` on the range where simulation already exists (`d < 12000`).
2. Only then walk to `d ~ 10^6`, resolving each of the ~15 branch points.
3. **Do not** re-propose a cycle certificate or an exhaustive state-graph
   sweep — both are closed by Result 3.

## Commands

```bash
python experiments/orbit_cycle_structure.py --pretty \
    --out data/wedge/orbit_cycle_structure.json
```

Runs in ~3 min.

## Artifacts

- `data/wedge/orbit_cycle_structure.json` — the `f_vec` gate, the Floyd outcome
  on the real orbit, the full survival table, and the cycle-feasibility figure.

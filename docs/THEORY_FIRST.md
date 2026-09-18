# The theory-first loop

This repo is very good at producing finite exclusions and has produced no
theorems since the diagonal recursion. That is not a tooling failure. It is
what happens when the loop's only artifact is an experiment.

This document defines the loop whose artifact is a **proof obligation**, and
the place computation holds inside it.

## Why the old loop cannot finish

All three prizes, in their believed direction, are lower-bound statements: no
period exists, the density is exactly 1/2, the cost is at least `O(n)`.
Computation can do exactly two things against a statement of that shape.

1. Find the unlikely **positive** — a period, a bias, a shortcut. Decisive, and
   the only outcome compute can deliver on its own.
2. Accumulate **finite exclusions**, which never compose into an asymptotic
   proof. `s*(64)=15` is a real theorem about 64 bits and says nothing about
   bit 65.

Almost all of this repo's compute has gone into (2). That ceiling is not moved
by a faster solver or a bigger GPU.

Meanwhile every result here that *did* generalise came from algebra, on paper:
left-permutivity giving `v_right = 1` exactly; Bernoulli invariance collapsing
Prize 2 onto the single seed; sideways determinism; the diagonal recursion and
its period-propagation lemma; the Reed–Muller bound that voided every
annihilator search at `w <= 22` before it ran. Each either *was* a theorem or
killed a large compute programme at zero cost.

Theory has outyielded computation here by a wide margin, and the loop is
overwhelmingly computational. The point of this document is to invert that.

## The loop

**1. Write obligations, not experiments.** An obligation is a lemma, the prize
implication it buys, and the cheapest thing that could refute it. If you cannot
state the implication, you do not have an approach. The format is
`queue/theory/obligations.json`.

**2. Gate on mechanism, free, before anything else.** `docs/theory/README.md`
section 4 lists twelve closed routes. Across them there are six mechanisms, and
`tools/theory_triage.py --list-mechanisms` names them. Every obligation must
say which mechanisms it is near and why it escapes them. Naming none is a
refusal: every proposal on this object is near at least one.

**3. Check the arithmetic, not the prose.** Where an obligation declares a
searchable class, the gate recomputes `log2|M|` against `n` from the repo's own
`experiments/counting_bound.py`. Prose can assert a search is fine; the
arithmetic cannot be argued with. The gate refuses `s <= 5` DFAOs at `n = 128`,
which is precisely the certificate retracted in 2026-08.

**4. Rank with Jev, then think.** Among survivors, which deserves the lead's
next hour? That is a typed choice over a closed set, costs about $0.00006, and
is far cheaper than reasoning over the whole ledger in your own tokens. It
ranks attention. It judges no mathematics, and a high probability is not
evidence for a lemma.

**5. Computation last, and only as a falsifier.** The SAT and GPU machinery
stops producing headline results and starts killing conjectured lemmas fast. A
lemma that survives a cheap refutation attempt is worth your thinking time; one
that dies cost you minutes instead of a session.

**6. Score the loop on conditional theorems, not experiments run.**

## Escaping the counting squeeze

The counting bound wants `log2|M| >= n` or a negative says nothing. The
forced-positive gate wants `log2|M| <= n` or a fit exists by dimension alone.
They meet at one point, so **any design whose evidence is a bare negative is
informative only there.** That single fact explains why the DFAO programme
needs a fresh 10x-harder instance for each extra state.

Two designs escape it, and the gate recognises both:

- **`evidence: "extrapolation"`.** Fit on `N` coefficients, then require the fit
  to predict `H` coefficients that took no part in it. No parameter count
  forces that. A relation that extrapolates is a finite shortcut; one that does
  not was overfitting, whatever its budget.
- **`evidence: "curve-shape"`.** The claim is the shape of a budget curve
  against its controls — plateau versus growth — not any single point. Needs at
  least three sizes, a null, and a positive control.

Both require controls, because an instrument that cannot detect structure
excludes nothing.

## The first instrument built this way

`experiments/algebraic_relation.py`. Christol's theorem says a sequence over
`F_2` is 2-automatic exactly when its generating function is algebraic over
`F_2(x)`. So the question the repo answers by SAT at `n = 64` is the same
question as *does a bounded-degree algebraic relation exist*, and that is a
kernel over `F_2` rather than a SAT instance.

What that buys, measured:

| | DFAO via SAT | algebraic relation |
|---|---|---|
| prefix reached | `n = 64` | `N = 1024` (curve), `N = 8192` (bare negative) |
| model class | 14 states (`log2|M|` 120.6, **clears** `n=64`) | 289 coefficients (**does not** clear `N=8192`) |
| deciding instance | 353 s solve, 2.26 GB proof, 598 s check | under a second |
| positive control | Thue-Morse `s*=2` | Thue-Morse `C*=12`, plateaus over 6 lengths |

**Read that second column carefully.** The `D, E <= 16` negative at `N = 8192`
is a *vacuous* negative by this repo's own gate: `log2|M| = 289 < 8192`, so a
random string gives the same answer. The DFAO row it sits beside is not vacuous
— 120.6 against `n = 64` clears the bound. A bigger `N` at a fixed budget is
**not** a bigger class, and comparing the two as if it were is the mistake this
document exists to prevent. What the algebraic route actually buys is the
`C*(N)` *curve*, where the budget is allowed to reach `N` and the claim is the
curve's shape against a null — not the bare negative at either length.

The positive control is the strongest available: the instrument independently
recovers the textbook relation `(1+x)^3 f^2 + (1+x)^2 f + x = 0`, returning
`D=2, E=3`, and that budget stays flat from `N=256` to `N=8192`, so detection
power does not decay with length.

Result to date: the center column admits no algebraic relation with
`D, E <= 16` on prefixes to 8192 bits, and is indistinguishable from the random
null while Thue-Morse separates cleanly.

**This is still a finite-prefix exclusion.** It does not prove
non-automaticity; no finite `N` can. And per the warning above, that bare
negative does not clear the counting bound, so on its own it is vacuous. What
the route buys is the `C*(N)` curve — a quantity that is always defined, read
against a 7-seed null with a working positive control, for a fraction of the
compute a single DFAO instance costs — and the fact that the question now sits
in transcendence over `F_2(x)`, a field with real proof machinery, rather than
in DFAO state counts, which has none.

## Running it

```bash
python experiments/algebraic_relation.py --self-test        # controls first
python tools/theory_triage.py --list-mechanisms
python tools/theory_triage.py queue/theory/obligations.json \
    --policy jev --out runs/theory/triage-NNN
```

`--policy fixed` takes the first survivor and makes no model call, which is the
baseline any claim of selector benefit must beat.

# Experiment Log

- Date: 2026-09-03
- Title: **C2 — Algebraic Annihilator Search over `w`-bit Windows (Problem 2)**
- Claim Level: **Certificate** — the rank computation is exact over GF(2), with
  no sampling threshold, and the negative is admissible under the Admission
  Rule plus a sharper class-specific gate derived below.
- Run date: 2026-09-03
- Artifact: `data/prize/2026-09-03-algebraic-annihilator.json`
- Verifier: `python experiments/algebraic_annihilator.py --self-test`

## Goal

Item C2 of `docs/STATUS.md`. Prize Problem 2 asks for a shortcut to the `n`-th
center-column bit. Experiment S closed the *linear* form of that question: the
center column has maximal linear complexity `L(n) = n/2`, so no LFSR generates
it. C2 asks the next question up the degree ladder, which the ledger records as
not yet asked.

The motivation is that Rule 30's own update is degree 2 in ANF,

```
a(t+1, i) = a(t,i-1) XOR a(t,i) XOR a(t,i+1) XOR a(t,i)*a(t,i+1)
```

so low-degree GF(2) structure is exactly the kind a shortcut might inherit. It
is also the kind of structure the neural experiments are constitutionally blind
to — I/K/L fail on `s[i-13] XOR s[i-27]` while learning period-31 instantly.

Following standard algebraic cryptanalysis (Courtois–Meier), we look for an
**annihilator**: a nonzero polynomial `f` of degree `<= d` over the `w` bits of
a sliding window with

```
f(s[i], s[i+1], ..., s[i+w-1]) = 0   for every position i.
```

Such an `f` is a constraint the stream satisfies everywhere — a shortcut
candidate. Its absence, at parameters where absence is not automatic, is a
negative that constrains Rule 30.

## Setup

The monomial basis of degree `<= d` in `w` variables has
`D = sum_{k<=d} C(w,k)` elements. Evaluating every monomial at every observed
window gives a matrix `M` with `D` columns, and `f` exists iff `M` has a nonzero
right kernel. The whole question is the GF(2) rank of `M`.

- Input: `data/golden/center_col_golden_10M.bin` (10,000,000 bits, bit mean
  0.500222). The golden reference is **tracked in git**, so this experiment runs
  on a fresh clone and does not wait on A3.
- Grid: `w in {20, 22, 24, 26, 28, 32}`, `d in {2, 3}`. A degree-4
  extension is in flight; see *Next step*.
- Rank is computed by bitset Gaussian elimination over `uint64` words.

Rank is at most `D`, so `D + 64` distinct windows settle the question. **Full
rank on a subset proves full rank on the whole stream**, so subsampling does not
weaken the negative; only a deficiency needs the full stream, and gets it.

## The counting bound, and why this class needs two gates

`CLAUDE.md` rule 1 requires the counting bound before any "searched class `M`,
found no fit" experiment. The Admission Rule's form is `log2|M| < n`. Here
`|M| = 2^D`, so `log2|M| = D`, and the constraint count `n` is the number of
distinct windows. That gives the familiar gate: **`D < n_distinct`**, or the
kernel is nonempty by dimension alone and any sequence produces the "positive".

This class also fails in the *other* direction, which the DFAO class does not,
and that gate turned out to be the binding one:

> An annihilator must vanish on every observed window. The minimum distance of
> the Reed–Muller code `RM(d, w)` is `2^(w-d)`, so a nonzero polynomial of
> degree `<= d` takes the value 1 at least `2^(w-d)` times and therefore
> vanishes **at most `2^w - 2^(w-d)` times**. If the stream shows more distinct
> windows than that, no annihilator can exist *whatever the sequence is*.

This is a theorem, not an estimate. Measured coverage against it:

| `w` | distinct windows | `2^w - 2^(w-2)` ceiling | verdict at `d=2` |
|---:|---:|---:|---|
| 20 | 1,048,499 | 786,432 | **VACUOUS — forced negative** |
| 22 | 3,807,262 | 3,145,728 | **VACUOUS — forced negative** |
| 24 | 7,531,795 | 12,582,912 | informative |
| 26 | 9,290,394 | 50,331,648 | informative |
| 28 | 9,815,389 | 201,326,592 | informative |
| 32 | 9,987,988 | 3,221,225,472 | informative |

At `w <= 18` the extreme case holds outright: every one of the `2^w` windows
occurs, so only the zero polynomial can vanish on them. A "no annihilator"
headline at `w <= 22` is a restatement of "the stream is varied enough" and says
nothing about Rule 30. `experiments/counting_bound.py` gained
`annihilator_verdict` and `max_zeros_of_degree`, and the experiment refuses to
report a result at parameters that fail either gate.

**This gate caught a real error mid-run.** The first pass reported rank 184
against `D = 211` at `w = 20` — an apparent shortcut. Two things were wrong. The
subsample was an evenly spaced slice of `np.unique`'s *sorted* output, so
consecutive rows shared high-order bits and manufactured rank deficiency; that
is now a seeded random permutation. And `w = 20` should never have been searched
at all: it sits above the Reed–Muller ceiling, so its negative was forced anyway.
The candidate did not survive full-stream verification, so nothing false was
recorded — but only because verification was in the path.

## Result

Full rank at every informative parameter. **No annihilator of degree `<= 3` over
windows of up to 32 bits.**

| `w` | `d` | `D` | rank | deficiency | verdict |
|---:|---:|---:|---:|---:|---|
| 24 | 2 | 301 | 301 | 0 | no annihilator |
| 24 | 3 | 2,325 | 2,325 | 0 | no annihilator |
| 26 | 2 | 352 | 352 | 0 | no annihilator |
| 26 | 3 | 2,952 | 2,952 | 0 | no annihilator |
| 28 | 2 | 407 | 407 | 0 | no annihilator |
| 28 | 3 | 3,683 | 3,683 | 0 | no annihilator |
| 32 | 2 | 529 | 529 | 0 | no annihilator |
| 32 | 3 | 5,489 | 5,489 | 0 | no annihilator |

`w <= 22` was refused at every degree by the Reed–Muller gate and is reported as
`skipped_vacuous` rather than as a clean negative.

Every row above is in the artifact. The degree-4 grid is a separate run and is
deliberately **not** summarised here until its artifact exists — `w = 24, d = 4`
(`D = 12,951`) has come back full rank, but a partial grid is not a result.

## Controls

A negative control that passes while testing nothing is worse than none, so the
load-bearing control here is a **positive** one.

- **Positive control — must find a relation.** A degree-2 NFSR of order 20,
  `s[i+20] = s[i] XOR s[i+3] XOR s[i+17] XOR (s[i+1] AND s[i+11])`, so the
  polynomial `x20 + x0 + x3 + x17 + x1*x11` annihilates every 21-bit window by
  construction. The search **finds it** and confirms it holds at every position
  of the full stream. A linear feedback would have exercised only the degree-1
  part of the basis, which Experiment S already covers; the quadratic term makes
  the control test the machinery this experiment actually depends on. The order
  is 20 rather than something small because both gates must clear — short
  registers produce too few distinct windows and trip the dimension gate.
- **Random control — must find nothing.** A seeded random stream of the same
  length returns full rank at every informative `w`, and is refused as vacuous
  at `w <= 22` exactly as Rule 30 is.

Note that the random control returning the same negative as Rule 30 at `w >= 24`
is **not** reassurance — per the Admission Rule it would be a red flag if the
counting bound had not been cleared independently. It has been: the Reed–Muller
headroom at `w = 24` is 5,051,117 windows, so a Rule 30 stream carrying a
degree-2 relation would have had room to show one.

## Interpretation

The algebraic route to Problem 2 is now closed two degrees further than it was.
Combined with Experiment S:

- degree 1 (LFSR / linear recurrence): closed, maximal linear complexity.
- degrees 2–3 over windows to 32 bits: closed, full monomial rank.

This is a genuine constraint on Rule 30 rather than a restatement of its
randomness, because the gate proves a relation *could* have been detected at
these parameters had one existed.

It is **not** a proof that no shortcut exists. The searched class is windows of
consecutive center-column bits under a degree bound. A shortcut could still live
in: higher degree; wider windows; non-consecutive bit selections; relations over
the space-time field rather than the column; or any non-polynomial form.

## Next step

**Degree 4 is running** over `w in {24, 26, 28, 32}` and will land as a separate
artifact and a ledger amendment. `D = 41,449` at `w = 32`, so it costs roughly
30x the `d = 3` grid; `w = 24` is already confirmed full rank.

Beyond that, the cheapest strengthening is width, not degree. The Reed–Muller headroom grows
as `2^w` while `D` grows polynomially, so wider windows are gated far more
loosely than deeper ones — but window codes are packed into `uint64`, capping
`w` at 32. Lifting that cap to multi-word codes is a contained change and would
let `w` run to 64 at `d = 2` (`D = 2,081`), a much larger class than anything
tested here at a comparable cost.

Second, the 46M stream would raise every distinct-window count and with it the
gate headroom, but it does not change which parameters are informative — the
ceiling is set by `w` and `d`, not by stream length. It is a lower priority than
width for this experiment, though A3 still blocks it.

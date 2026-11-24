# 2026-11-24 — period one machine-checked in Lean; what did not extend to period two

## Goal

Move something on the prize ladder from "argued" to "proved", after a session
of finite computations that each came back "as random". The only rung that is
actually settled, period-one exclusion, was held at Proof candidate because it
rested on an unrefereed preprint read by one model lineage. A proof checker
removes that caveat; a model cannot.

## Result — `formal/Rule30/PeriodOne.lean` (Lean 4.19.0, core only)

Three theorems, no `sorry`, standard axioms only:

- `no_eventually_constant_column`: for every finite-support, nonzero initial
  row and every column `c`, the column is not eventually constant.
- `centre_column_not_eventually_constant`: for the single black cell, both
  colours occur in the centre column at arbitrarily late times.
- `centre_column_gap`: for every `T`, both colours occur in the centre column
  at some time in `[T, 4T+2]`.

Verifier: `LEAN=/path/to/bin/lean python tools/verify_lean.py`. The file also
checks, by `decide`, that the formal seed's centre column starts
`1,1,0,1,1,1,0,0,1,1,0,0,0,1,0` (OEIS A051023), so the object proved about is
the prize object.

### The argument (simpler than the fiber classification)

- **Eventually one.** If column `c` is 1 at every time then
  `1 = a(t,c-1) XOR 1` forces column `c-1` to 0; then `0 = a(t,c-2) XOR 1`
  forces column `c-2` to 1; and so on. The row contains the infinite
  checkerboard, contradicting finite support.
- **Eventually zero.** If column `c` is 0 at every time, the right neighbour
  obeys `s' = s OR u` and the left neighbour equals the right neighbour. If the
  latch never fires, columns `c, c+1` are zero forever and zero propagates both
  ways, so the row is zero. If it fires, column `c-1` is eventually one, which
  is the first case. So the zero case **reduces** to the one case; no fiber
  classification is needed.
- **Window.** The same steps inside a finite horizon: a run of ones from `T`
  longer than `T+2` pushes the checkerboard outside the light cone; a run of
  zeros either leaves the latch unfired for `T` steps (then the black left edge
  of the cone is contradicted) or fires it by time `T` (then the checkerboard
  argument applies from there). Total `4T+2`.

### Scope and credit

The theorem is known: eventually-one is Jen (1986), eventually-zero is Condrey
(arXiv:2609.09431, 2026), whose sharp horizons give a window of about `T+2`
rather than `4T+2`. What is new here is the machine check and the reduction.
Consequence for Prize 2, stated at its true size: each colour occurs at least
about `log4(n)` times in the first `n` centre bits. That is a proved lower
bound where the repo previously had only a 10M-bit observation, and it is
nowhere near equidistribution.

## What was tried on period two, and why it is not in the theorem

1. **Left-only relaxation.** With an alternating centre the left half evolves
   on its own, and the centre update forces the nearest left cell to be 1 at
   every odd time - a condition on the left half alone. If no finite left half
   could satisfy it forever, period two would fall without the right half.
   `experiments/period2_left_only_horizon.py`: over every left half of exact
   radius `R <= 22` the longest survival is `K_max(R)` =
   3,1,2,2,4,4,4,8,7,8,7,14,11,13,15,12,15,16,14,18,17,20 double-steps. That
   is `R + O(1)` with irregular scatter - exactly the maximum of `2^R`
   independent geometric lifetimes - and the extremisers share no pattern.
2. **No sharp law.** For constant targets the certified minimum radius in
   `runs/period-horizon-2026-10-14/results.json` is exactly `L-1`/`L-2`
   alternating by parity, which is why period one has a proof. For the
   alternating target `L - R_min` wanders (3,4,4,3,...,15,7,8,...,9,10 over
   `L=8..64`) like the random-word controls. There is no extremal family to
   induct on.
3. **Forced cells thin out.** Under an alternating centre (column 1 at even
   times written `Y_k`): column -1 is 1 at odd times and `NOT Y_k` at even;
   column -2 is `Y_k` on the time pair `(2k-1, 2k)`; column -3 is the
   complement of column -2 one step later; column -4 is 0 at every even time.
   Beyond that the columns are genuine functions of `Y` with growing
   look-ahead, so the rigid checkerboard that drives the period-one proof is
   replaced by one free bit per double-step.

Conclusion: period two is not a harder instance of the period-one argument.
Every finite quantity measured behaves like one coin flip per step, so a proof
needs an invariant nobody has, not a longer computation. Do not re-run 1-2.

## Prize 3 side-check — digit-Fourier profile (null)

`experiments/digit_fourier_profile.py`: `n -> c_n` on 23-bit indices has max
Walsh coefficient z = 5.70 (seven-seed null 5.12..5.73, expected 5.65), every
digit sensitivity in 0.4996..0.5007, and a generic ANF. So the centre bit is
uncorrelated with every parity of the binary digits of `n`. Statistical null;
complements the exact DFAO lower bound of 2026-11-23.

## Interpretation

The session's honest balance: one statement moved from preprint to
machine-checked theorem with an explicit quantitative corollary; no open case
moved. Prize 1 still needs period two and beyond, Prize 2 needs a density and
has a logarithm, Prize 3 has no proof technique at all.

## Next Step

Formalise Condrey's sharp horizon (`T+2`) to tighten the window, and port the
alternating-centre column identities in item 3 into `formal/` so any future
period-two argument starts from checked lemmas (PR 50 holds overlapping ones).

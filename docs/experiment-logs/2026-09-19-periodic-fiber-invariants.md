# 2026-09-19 — Periodic-fibre invariants for Prize 1

## Goal

Replace finite alternating-prefix scans with an exact inverse-fibre quantity
that has an explicit bridge to eventual period two, then search small induction
languages for a uniform bound.

Nothing here excludes period two or settles a Wolfram prize.

## Exact reduction

Fix a periodic centre word, its phase, and an initial right half.  Because Rule
30 is left permutive, the desired centre bit at time `d` uniquely determines
the initial bit at coordinate `-d`, without changing any earlier trace bit.
Write `L_d(r)` for this forced bit when the right half is `r`.

For support radius `w`, a row realizes the target through time `T` exactly when

```text
L_(w+1)(r) = ... = L_T(r) = 0.
```

Consequently the maximum alternating-prefix length at radius `w` equals

```text
E(w) = max over right words r of the first d>w with L_d(r)=1.
```

The equality is just triangular uniqueness; it is a reformulation, not a
period-two theorem.  A proved finite upper bound for `E(w)` for every `w` would
exclude an eventually alternating centre for every finite configuration.

## Implementation and controls

`experiments/periodic_trace_fiber.py` constructs the fibre for any periodic
binary word and any left-permutive elementary CA.  Rule 30 steps are checked
against the repo's independent naive stepper.  Several words and every phase
are reconstructed and forward-verified.

Rule 90 is the negative control: with target `0`, right bit `+1 = 1`, and Rule
90, the inverse left half is exactly `1 0 0 ...`.  Thus a future argument that
uses only left permutivity would falsely exclude a real finite periodic fibre
and is rejected.

`experiments/periodic_fiber_invariants.py` exhausts every right word, records
`E(w)`, freezes radii `1..10` as training data, and treats larger radii as
holdout.  Construction depth was 64; every row escaped before the boundary, so
none of these maxima is right-censored.

## Exact bounded result

For alternating traces, combining both phases gives:

```text
w:     1  2  3  4  5  6  7   8   9  10  11  12  13  14  15  16  17  18
E(w):  7  7  7  7  9 10 10  15  17  17  17  17  17  20  22  26  26  26
```

The first ten values independently reproduce the earlier horizon implementation.
The run through radius 16 is
`runs/theory/periodic-fiber-invariants-k16-d64.json`; radii 17 and 18 are in
`runs/theory/periodic-fiber-invariants-k18-supplement.json`.

The tight slope-one law fitted on training data, `E(w) <= w+8`, fails on the
holdout at `w=16`, where `E(16)=26`.  The next simple envelope,
`E(w) <= 2w+5`, survives through `w=18`.  This is a proof target only.  The
survivor counts and jump locations are irregular, and nothing here supports
extrapolating the constant `5`.

## Failed induction languages

Two common spatial recurrences were fitted only on training depths and checked
on later depths and larger right supports:

- no phase-aware affine recurrence exists with memory at most 24 and depth
  modulus in `{1,2,4,8}`;
- no phase-aware quadratic recurrence exists with memory at most 12 and depth
  modulus in `{1,2,4}`.

These are exact counterexamples to those templates, not evidence that no
recurrence exists.  They do show that the forced tail does not collapse to the
obvious low-memory expression suggested by Rule 30's quadratic local rule.

## Interpretation

The inverse-fibre formulation is the useful result: it makes every candidate
state exactly what must be proved and gives immediate finite counterexamples.
The measured horizon still behaves like a maximum over `2^w` complicated
constraints.  Scaling `w` again would mainly double the cost and add one point.

The next justified work is a width-uniform argument: a transfer state,
renormalisation, or well-founded potential showing that the all-zero forced
tail cannot persist.  `E(w) <= 2w+5` is the sharpest current candidate to attack
or falsify algebraically.  Do not call it a law and do not launch larger radii
without a proposed proof mechanism that predicts them.

If proved, this would exclude eventual period two only.  Every period at least
three would remain open, so it would be a genuine first new rung toward Prize 1,
not the full prize.

## Follow-up: the right half can be eliminated

The fibre calculation was subsequently compressed to an exact autonomous map
of the initial left half.  Encode the cells `-1,-2,...` at a centre-zero phase
as the low-to-high bits of a nonnegative integer `n`.  Direct substitution in
Rule 30 gives

```text
T_b(n) = (n >> 1) XOR (n OR ((n << 1) OR b)),
G = T_1 o T_0.
```

An alternating centre requires the cell at `-1` to be one at every centre-one
phase.  Exactly, this is

```text
G^k(n) mod 4 in {1,2}  for every k >= 0.
```

Thus period-two exclusion is reduced to proving that every ordinary integer
eventually exits this residue set.  The reduction is a theorem; termination is
still open.  It is stronger and cleaner than the right-fibre formulation
because it makes no assumptions about, and keeps no state from, the right half.

`experiments/alternating_left_transfer.py` independently checks the integer
identity cell by cell.  Exact maximum first-failure macrotimes `Q(w)` over all
`0 <= n < 2^w`, for `w=1..20`, are

```text
3,3,3,3,4,4,4,8,8,8,8,14,14,14,15,15,15,16,16,18.
```

There are no right-censored cases.  The candidate `Q(w) <= w+2` survives this
range only.  Proving it would imply a linear period-two exclusion, but any
finite bound `Q(w)<infinity` for each `w` is enough.

### Survivor-set compression and its ceiling

`experiments/periodic_fiber_survivor_bdd.py` retains the exact Boolean set of
right halves surviving every forced-zero constraint, rather than just its
cardinality.  Through support 16, the hardest last-nonempty sets have natural-
order ROBDDs of at most 81 nonterminal nodes and maximum layer width 14.  This
is evidence that proof mining may admit a compressed state; it is not evidence
that the survivor chain must empty.

Rule 15 is now the decisive alternating negative control.  Its rule is
`F(l,c,r)=1 XOR l`, so an all-zero left half produces an eternal alternating
centre from finite support.  Its survivor function is constant true and its
ROBDD has zero nonterminal nodes.  Therefore diagram compactness, triangular
inversion, and left permutivity alone cannot prove the Rule 30 result.  A valid
abstraction still needs a Rule-30-specific ranking or a proof that its accepting
subgraph has no cycle.

The tempting bivariate generalization `E_phi(ell,r) <= ell+r+5` is false.  Exact
counterexamples include phase zero with `(ell,r)=(20,5)`, whose next forced one
is at depth 35 rather than at most 30, and `(52,1)`, whose next forced one is at
62 rather than at most 58.  The surviving diagonal envelope is not explained
by a uniform bounded-gap lemma.

The next proof-facing targets are fixed-weight renewal for the integer map,
zero-tail elimination, and shiftable SAT clauses.  The detailed route and stop
rules are in `docs/theory/prize1-attack-map.md`.

### Cone-charge conjecture

Sideways reconstruction gives a stronger, one-free-bit-per-macrostep search.
Let `s_k` be the right neighbour at even time `2k`; all `2^m` words `s` induce
the first `2m` bits `e` of the initial left half.  Exact minimization over all
`2^22` words, checked against directly evolved spacetime rows, gives even-prefix
minimum weights

```text
m:       1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22
min wt:  1 1 1 2 2 2 2 2 3  3  3  4  4  4  4  4  5  5  5  6  6  6
```

Every prefix through spatial depth 44 satisfies the conjectured bound

```text
wt(e_1...e_N) >= floor((N+6)/8).
```

At `N=2m`, this is the cone-charge lemma
`wt(e_1...e_2m) >= ceil(m/4)`.  A proof would exclude an eternal alternating
centre from every finite configuration and settle period two.  The bounded
result is only a Certificate.  The next action is not `m=23`: synthesize a
bounded local cut potential whose verified Rule 30 tile inequalities telescope
to the charge bound.  A small exact transducer should not be assumed; bounded
residual signatures grow rapidly and fixed input memories already collide.

A second tempting shortcut is also falsified.  Keep a fixed-height vertical
window of two adjacent temporal columns, reconstruct the next column wherever
the window determines it, and allow the forgotten top bit to branch.  For every
height 2 through 10 this relaxed graph reaches the all-zero pair, which has a
negative self-cycle for the proposed weight charge.  The shortest distances are
`2,5,6,7,8,9,10,15,16`.  This does not refute a local potential; it shows that
the cut must follow the slanted dependency characteristic (or retain a boundary
credit), rather than repeatedly discarding the top temporal constraint.

The same left-only construction extends to a period word `u`: compose the
maps `T_(u_j)` over one period and impose
`bit_0(n_j)=1 XOR u_(j+1)` at every centre-one phase.  Every nonconstant cyclic
word has a `1 -> 0` latch, so this supplies a uniform framework for all of Prize
1.  As a bounded falsifier, all primitive words of periods 2 through 8
(including rotations) and all initial left halves below `2^10` exited within a
200-period cap; maximum first-failure times by period were
`9,9,8,8,9,8,11`.  This is not an all-period result.  Its value is that the
same kind of cone-charge theorem could in principle scale by period word,
without first reconstructing the right half.

## Reproduce

```bash
python -m unittest tests.test_periodic_trace_fiber \
  tests.test_periodic_fiber_invariants tests.test_alternating_fiber_prefix

python experiments/periodic_fiber_invariants.py \
  --train-max 10 --holdout-max 16 --depth 64 \
  --out runs/theory/periodic-fiber-invariants-k16-d64.json

python experiments/periodic_fiber_invariants.py \
  --verify runs/theory/periodic-fiber-invariants-k16-d64.json

python experiments/periodic_fiber_invariants.py \
  --verify runs/theory/periodic-fiber-invariants-k18-supplement.json

# Full extension; substantially slower because every added radius doubles the
# number of right halves.
python experiments/periodic_fiber_invariants.py \
  --train-max 10 --holdout-max 18 --depth 64

python experiments/alternating_left_transfer.py \
  --max-width 20 \
  --out runs/theory/alternating-left-transfer-w20.json

python experiments/periodic_fiber_survivor_bdd.py \
  --max-support 16 --depth 64 \
  --out runs/theory/periodic-fiber-survivor-bdd-k16-d64.json

python experiments/alternating_left_weight.py \
  --macrosteps 22 \
  --out runs/theory/alternating-left-weight-m22.json

python experiments/alternating_left_transfer.py \
  --max-width 20 --scan-period-max 8 --scan-width 10 --scan-limit 200 \
  --out runs/theory/periodic-left-transfer-p8-w10.json
```

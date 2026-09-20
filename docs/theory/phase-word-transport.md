# One-event transport and threshold words for arbitrary centre periods

Date: 2026-09-20. This note extends the physical `11` exclusion beyond
alternation, while identifying exactly what information an event encoding
must retain. It supplies no initial-weight charge, no period exclusion, and
no claim of literature novelty.

## 1. A zero run has an exact threshold profile

Encode the right half with the nearest cell in bit zero and write

```text
R_b(x) = ((x << 1) | b) XOR (x | (x >> 1)).
```

For every `m>=0`, the nearest cell after `m` zero-boundary steps is

```text
bit_0(R_0^m(x)) = OR_(0<=j<=m) bit_j(x).                 (1)
```

This holds for finite or infinite right halves. To prove it, if the initial
first `m+1` cells are zero, their zero cone gives output zero. Otherwise let
`j<=m` be the position of the first one. That one advances left by one cell
per step through the preceding zero cells, because `F(0,0,1)=1`, while cells
strictly to its left remain zero until arrival. It reaches the nearest cell
at step `j`. Under a zero centre, that cell is then permanently one because
`F(0,1,z)=1`. The case `j=0` is included.

For an event block with centre boundary word `0^a1`, `a>=0`, define
`S_a=R_1 o R_0^a`. Since the low bit of `R_1(y)` complements that of `R_0(y)`,
(1) gives the exact identity

```text
bit_0(S_a(x)) = 1 - OR_(0<=j<=a+1) bit_j(x).             (2)
```

The case `a=0` is essential: consecutive centre ones are allowed.

## 2. Every centre history with infinitely many ones has a golden-mean factor

Let `t_j` be consecutive times when the prescribed centre equals one. Sample
right halves at rows **immediately after** these boundary updates:

```text
X_j = right_half(t_j+1),  a_j = t_(j+1)-t_j-1.
X_(j+1) = S_(a_j)(X_j),  s_j = bit_0(X_j).
```

Equation (2) implies `s_j=1 => s_(j+1)=0`. Thus the sampled word avoids `11`
for every such centre history, including arbitrary zero-run lengths and
consecutive ones. It is not specific to period two.

For a nonconstant cyclic centre word, the finite ordered list of gap lengths
`a_j` repeats and contains at least one positive entry. The event labels
must retain this ordered list; replacing all gaps by one silently changes
the dynamics. For a finite-configuration eventual-periodicity question,
one may restart immediately after a centre-one event without losing finite
support. Neither this rephasing nor the golden-mean factor proves a charge.

## 3. Threshold labels recover the phases hidden by event sampling

Define `nu(x)` as the position of the first right one, with `nu(0)=infinity`,
and retain the finite label

```text
lambda_j = min(nu(X_j), a_j+2),   in {0,...,a_j+2}.
```

At each row within the block, after `h` zero steps and before its final
one-boundary update, (1) gives

```text
r_(j,h) = 1[lambda_j <= h],                    0 <= h <= a_j.
s_(j+1) = 1[lambda_j = a_j+2].
```

Consequently successive threshold labels satisfy the exact necessary seam
condition

```text
1[lambda_(j+1)=0] = 1[lambda_j=a_j+2].         (3)
```

For a fixed periodic centre word, these labels form a finite alphabet with a
periodic event phase. They recover the entire nearest-right observation,
not merely its sampled bits. They do **not** form a complete presentation of
physical right traces: (3) forgets shared higher right predecessors.

The centre equation also gives the complete nearest-left word from the
thresholds. At the zero phases and at the final one phase respectively,

```text
l_(j,h) = 1[h=a_j-1] XOR 1[lambda_j<=h],       0 <= h < a_j,
l_(j,a_j) = 1[a_(j+1)>0].
```

The first range is empty when `a_j=0`. In the second equation the following
centre is zero exactly when the next gap is positive. All deeper left
columns then follow by the usual exact sideways inversion. Thus an
all-period charge search can use an explicit phase/threshold input alphabet
without discarding the masked within-run observations. Its spatial cuts still
grow; a finite input alphabet does not supply a bounded cut automaton.

There is a small exact collision explaining why the thresholds matter.
For gap `a=2`, finite right halves `2` and `4` both have sampled endpoints
`0 -> 0`, but their nearest-right histories over the three updates are
`0110` and `0010`. The hidden difference occurs at a centre-zero phase and
changes the required nearest-left history. Gap labels plus sampled `s` are
therefore insufficient outside alternation.

## 4. The bounded-zero rank does not generalise with the golden-mean factor

The named candidate that every fixed nonconstant period supplies a uniform
bound on runs of zero event samples is false, even for a finite initial right
half. Prescribe centre word `001` and apply `S_2`. The following set of low-four
right prefixes is universally forward invariant:

```text
K = {2,4,6,8,10,12,14}.
```

Its exact projected successors, allowing all three outer dependency bits,
are:

| Low four bits | Successor low four bits |
|---|---|
| 2 | 2, 10, 14 |
| 4 | 4, 12 |
| 6 | 4, 14 |
| 8 | 2, 10 |
| 10 | 14 |
| 12 | 8, 12 |
| 14 | 4, 12 |

Each successor is even and nonzero modulo sixteen. Three updates have radius
three, so the seven initial right bits checked in this table contain every
influence on the four outputs. Universal closure therefore proves invariance
for all future events, independently of the changing higher tail.

Starting with the finite empty right half gives the exact integer steps
`0 -> 1 -> 12`. Thereafter it remains in `K`, so the sampled word is
`0,1,0,0,0,...`. This is an infinite-duration counterexample, proved by finite
universal closure, not a long censored simulation. It does not assert that the
full nearest-right trace becomes periodic: the threshold positions can vary
while every sample remains zero.

This is a physically coherent *forced right half*, not a finite full-row
counterexample. The reconstructed left half has not been shown finite. It
refutes transfer of the alternating `00000` exclusion to arbitrary centre
periods; it does not refute the all-period charge goal.

## 5. What the reduction leaves to prove

For each cyclic gap list, a sufficient route is a divergent initial-left
weight bound on every reconstructed threshold word satisfying (3), or on a
stronger sound language retaining additional right predecessors. Demand can
count `1 -> 0` events, equivalently positive gaps per event cycle. No uniform
coefficient over all periods is required.

The substantive missing step remains spatial transport with correct
initial-one accounting, inherited cut information, and uniform endpoint
bounds. The positive reduction here makes the all-period boundary language
explicit. The period-three trap prevents falsely importing the alternating
right rank as the next step.

Run `python experiments/phase_word_transport.py`. The independent Wolfram
truth-table cones check (1), (2), threshold profiles, and left equations for
zero-run lengths through ten; the symbolic proof supplies arbitrary-run
scope. All 56 low-four plus outer-bit cones establish the finite invariant
used in the induction. These are executable mathematical checks, not a
proof-assistant formalisation.

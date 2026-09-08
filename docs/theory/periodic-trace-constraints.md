# Periodic traces, seed anchoring, and time-shift defects

This note develops proof targets for official Prize Problem 1. The identities
below are elementary consequences of left-permutivity, not claims of new
literature results. None proves nonperiodicity of the single-seed center.

## 1. A control that any periodic-strip search must pass

Use `x_i(t+1) = x_{i-1}(t) XOR (x_i(t) OR x_{i+1}(t))`.
For each nonnegative `t`, there is a Boolean function `G_t` such that

```
x_0(t) = x_{-t}(0) XOR G_t(x_{-t+1}(0), ..., x_t(0)).
```

The extreme left initial bit reaches the output along the unique path moving
right at every step. At each step it enters as a pure XOR. Inductively, the
left child contains this bit exactly once as an XOR term; neither other child
can depend on it because their light cones start strictly to its right. This
proves the displayed identity, including coefficient one, at every depth.

Fix every positive initial coordinate to zero. Given any target trace
`c_0,...,c_T`, choose `x_0(0)=c_0`. Having chosen through `x_{-t+1}(0)`, evaluate
the time-t center with `x_{-t}(0)=0`, and choose `x_{-t}(0)` as that answer XOR
`c_t`. This changes no earlier output. Induction gives a unique initial word
on `[-T,0]`, with zero extension outside, realizing the target through time T.

The choices are compatible when T increases: later steps never revise earlier
initial bits. Consequently **every infinite center trace** has a unique left
half-line initial condition when the positive half-line is fixed to zero.
That initial condition need not have finite support. Finite-support realizers
of longer and longer prefixes must not be mistaken for one finite-support
realizer of an infinite trace.

### Implication for the proposed search

A periodic center word with otherwise free initial conditions always has an
infinite spacetime realization. Its restriction is a witness in any finite
strip with open boundary conditions. A solver rejecting it has imposed extra
conditions (or has a bug). Spatially cyclic boundaries are extra conditions.

Even fixing the actual seed on `[-K,0]`, and zero on the entire positive
half-line, is insufficient: every extension of the seed's first `K+1` center
bits remains possible by choosing the still-free left tail. In particular,
one can prescribe any eventual periodic continuation after that prefix.

This does not rule out proofs using finite local lemmas. It rules out treating
a bounded initial patch as if it encoded all the zero cells of the seed. A
successful argument needs a uniform reason that the full seed cannot realize
the hypothesized tail, or must propagate its constraints to an unbounded
distance. The elementary finite-prefix counting gate is not the issue here:
`2^(T+1)` initial words realize exactly `2^(T+1)` traces, a bijection.

## 2. The exact defect equation

Let `b` and `y` be any two Rule 30 trajectories, and set `d_i(t)=y_i(t) XOR
b_i(t)`. Over GF(2), subtraction equals XOR and OR has polynomial form
`u OR v = u XOR v XOR uv`. Expanding the two update rules gives

```
d_i(t+1) = d_{i-1}(t)
           XOR ((1 XOR b_{i+1}(t)) * d_i(t))
           XOR ((1 XOR b_i(t)) * d_{i+1}(t))
           XOR (d_i(t) * d_{i+1}(t)).
```

For the time-shift comparison use `b_i(t)=x_i(t)` and `y_i(t)=x_i(t+p)`.
These are two trajectories of the same rule, so the equation applies without
an independence assumption. The background coefficients are essential;
defects do not form a closed three-input cellular automaton on their own.

If `d_i(t)=d_i(t+1)=0`, this reduces to

```
d_{i-1}(t) = (1 XOR b_i(t)) * d_{i+1}(t).
```

Thus a defect-free center forces the left defect to vanish at center-one
times, and equates the two neighbor defects at center-zero times. This is a
conditional identity, not proof that either neighbor is eventually defect-free.

## 3. What an inverse-seed witness adds (and does not add)

Construct a target agreeing with the true seed trace until a chosen onset s,
then repeating its observed block of length p. Let tau be its first mismatch
with the true trace, if one occurs in the computed horizon. Under the fixed
positive-half-line convention, triangular inversion produces

```
x_0(0)=1, x_{-1}(0)=...=x_{-(tau-1)}(0)=0, x_{-tau}(0)=1.
```

Proof: the common output prefix has a unique inverse initial prefix; at tau,
the different output forces the new input bit to flip. The remaining inverse
bits have no bearing on this first violation. Forward-simulating the modified
seed checks that the proposed periodic prefix is locally realizable, while
the true seed checks where it fails.

This witness is an interpretable form of an ordinary first mismatch, not a
stronger period exclusion. A finite scan over `(s,p)` cannot exclude later
onsets or larger periods. If no mismatch occurs before the horizon, the case
is unresolved, even if its first block is known exactly. There is no reason
to scale this probe toward the repo's existing millions-of-periods search.

## 4. Constant centers give a simple synchronization control

If the center is eventually one, its update forces `x_{-1}(t)=0` throughout
that tail. If it is eventually zero, its update gives `x_{-1}(t)=x_1(t)`.
Meanwhile `x_1(t+1)=x_1(t) OR x_2(t)`, so the right-neighbor bit is
nondecreasing: it changes at most once and then stabilizes. The left neighbor
therefore stabilizes too. This elementary argument applies to all backgrounds.
It supplies a positive control for the finite-strip synchronization tool.

For an alternating center, the simple monotonicity argument disappears. The
tool builds all phase-labelled strip states, applies the local rule to every
interior cell, and allows both boundary cells to change freely at every step.
It then checks recurrent strongly connected components for a p-step path
changing the left-neighbor bit, where p is the prescribed word length. A
return path closes each counterexample walk, making the violation repeat
indefinitely in the strip. If none exists, every infinite strip path eventually
has a p-periodic left neighbor: it eventually stays in one recurrent component.

A counterexample refutes only this strip implication. In particular, a neighbor
with a larger period is already a counterexample to period p; it need not be
aperiodic. Such a walk need not extend consistently outside the strip or be
reachable from the single seed. Adding seed-derived boundary restrictions is
a substantive extra hypothesis, not an implementation detail.

## 5. The remaining proof target

The useful target is a **seed-specific escape lemma**: for every p and onset s,
the hypothetical p-periodic continuation is incompatible with the full seed.
One possible formulation is that its inverse left tail must contain a one.
As stated, that is equivalent to the original problem, not a reduction in
difficulty. To make progress, derive a sufficient structural condition on a
period word that forces such a one, with a finite symbolic argument applying
to an unbounded family of periods/onsets.

The defect equation provides another route: show that a permanent defect-free
center would force a second adjacent defect-free column, using an additional
property proved for the seed orbit. The local equation alone does not provide
that property. Do not silently replace the missing implication by sensitivity,
randomness, finite-strip recurrence, or a sampled absence of counterexamples.

Background: the official [prize announcement](https://writings.stephenwolfram.com/2019/10/announcing-the-rule-30-prizes/)
discusses sideways reconstruction and the obstruction involving adjacent
periodic columns. The repo's [theory gate](README.md) already records
left-permutivity and the distinction between arbitrary and single-seed orbits.

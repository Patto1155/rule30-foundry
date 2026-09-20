# A quantitative periodic-boundary subclass of weak weight growth

This note isolates a sufficient subclass without assuming a universal linear
charge. It does **not** exclude an arbitrary alternating centre trace. The
mechanism is a quantitative form of the adjacent-periodic-column obstruction;
it is not claimed as a new general period-exclusion mechanism.

## Periodic even-right words force initial ones

Prescribe centre `c(t)=t mod 2`. Write `s_k` for its immediate right neighbour
at time `2k`. Let `X_j(t)` denote the reconstructed column at left depth `j`,
with `X_0=c`. Rule 30 gives

```text
X_(j+1)(t) = X_j(t+1) XOR (X_j(t) OR X_(j-1)(t)).
```

The actual right neighbour at odd times can be replaced by zero for this
reconstruction: `c(2k+1)=1` masks that bit in the OR. This replacement does not
claim that the auxiliary right column has a physical right extension. It
preserves every actual left column whenever an actual alternating orbit
exists.

**Lemma.** If the even-right word `s` is periodic with period `p>=1`, every
block of `4p` initial-left cells contains a one. Consequently its reconstructed
initial-left prefix weight satisfies

```text
W_N >= floor(N/(4p)).
```

Here `p` need not be the least period, and no `11`-free assumption is required.

**Proof.** Put `P=2p`. The auxiliary right column (even bits from `s`, odd bits
zero) and the alternating centre both have temporal period `P`. The sideways
recurrence therefore makes every reconstructed left column `P`-periodic. It
also ensures that their forward Rule-30 equations hold at all left sites.

Suppose `2P` consecutive initial-left cells are zero. After `t` forward steps,
the inner `2P-2t` cells of that interval remain zero, since `f(0,0,0)=0`.
Thus the middle two columns are zero at all times `0,...,P-1`. Their temporal
periodicity makes them zero at every time. Two adjacent identically-zero
columns force the next column to their right to be identically zero: the
update equation at the right member of the pair is

```text
0 = f(0,0,r(t)) = r(t).
```

Repeating toward the centre makes the alternating centre identically zero,
a contradiction. Partition an initial prefix into complete `2P=4p` blocks
to obtain the stated weight bound. QED.

The centre alternation is used both to retain only the sampled word `s` and
to contradict the zero pair. The zero-pair propagation is Rule-30-specific;
quiescence or permutivity alone does not prove this lemma for another rule.

## Finite-prefix version, without a lost boundary margin

Suppose the known word `s_0...s_(M-1)` has period `p<=M` in the ordinary finite
word sense: `s_(i+p)=s_i` wherever both indices exist. Extend its first `p`
bits periodically forever. The first `2M` reconstructed initial-left cells
are unchanged. Indeed the bit at left depth `j` depends only on right-column
times `0,...,j-1`, and therefore only on
`s_0,...,s_floor((j-1)/2)`; the odd right bits are masked as above.

Apply the infinite periodic lemma to this extension. There is no need to
subtract a temporal-cone margin:

```text
W_(2M) >= floor(M/(2p)).
```

Therefore, if an actual alternating prefix arises from an initial left half
of total weight `w`, and its first `M` even-right observations have period
`p`, then

```text
M < 2p(w+1).
```

Equivalently, a prefix `u^K` has `K < 2(w+1)`, independently of the length of
`u`. This bound is deliberately weak; it excludes long repeated prefixes
at fixed weight, rather than optimising the coefficient.

For `p=2`, the lemma supplies `W_N>=floor(N/8)` on the periodic-boundary
subclass. This is consistent with the sparse period-seven ether. It gives
no justification for applying coefficient eight to nonperiodic `s`.

## What this does and does not open

If a finite-support configuration had an eventually alternating centre and
its sampled even-right word were eventually periodic, restart at an even
phase after both transients. The left half still has finite support, but
the lemma would force infinitely many initial-left ones. This is impossible.
Thus any counterexample to period-two exclusion must have a genuinely
non-eventually-periodic **sampled even-right word**, not merely an aperiodic
full adjacent column.

The following two routes would complete a bridge, but neither is proved here:

1. Prove that forcing the centre to `01` makes the right-half return map
   `H=R_1 o R_0` eventually periodic at its low bit for every finite right
   initial condition. The local no-`11` rank and finite right-language
   checks do not establish this.
2. Prove enough repetition at a fixed temporal starting point to violate
   `M < 2p(w+1)`, or charge persistent failure of such repetition to growing
   initial-left weight. An arbitrary aperiodic word need not have long
   repeated prefixes, so generic compactness does not supply this bridge.

Allowing the repeated word to begin at progressively later times is not
sufficient: the left weight at each restarted time can grow. Likewise,
checking a larger set of periodic words adds no proof of the missing
nonperiodic case.

The general claim that every allowed boundary word forces unbounded initial
weight is still the desired exclusion in another form. The useful reduction
here is only the explicit repeated-prefix obstruction and its uniform bound
within each fixed-period subclass.

## Reproduction

```bash
python experiments/weak_weight_growth.py
```

The default run checks all 254 periodic words with period parameter one
through seven. It constructs exact circular temporal columns through spatial
depth `32p`, checks the claimed gap and weight bounds, independently replays
forward shrinking cones using the eight-entry elementary-rule truth table,
and compares different periodic extensions of a common boundary prefix to
check the `2M` locality convention. Both phases and words containing `11`
are included. No packed left or right transfer map is used.

This is a finite audit of the coordinate and Boolean identities supporting
the proof above. The all-period statement rests on the argument, not on the
number of audited periods. The default audit passed. No prize grade changes
follow.

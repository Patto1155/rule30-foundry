# Hostile review of interface composition and right coherence

## Verdict

The three notes contain four sound negative results and one explicitly open
positive route.  The negative results survive an independent check of the
phase conventions, dependency cones, and quantifiers.  They do not prove an
eventual-alternation exclusion.

This was a separately tasked review using the same model lineage.  Agreement
with the authors or their checkers is not independent mathematical evidence;
the reasons below are the proof obligations that actually carry each verdict.

| Claim | Verdict | Exact scope |
|---|---|---|
| The unrestricted sharp charge `q(n)<=4 wt(n)` | **Reject the bound; accept the counterexample.** | `4329177217` has weight 6 and `q=26`.  This is a theorem about the free-right-boundary left-cone quantity `q`, not a physically coherent alternating orbit. |
| A newly inserted bit has a fixed marginal delay allowance | **Reject.** | At the fixed position 42, inserting the missing bit into `m_R` makes the increase in `q` unbounded along odd `R`. |
| An isolated ether defect has a uniform post-arrival failure horizon | **Reject.** | The 2-adic inverse-limit survivor yields ordinary positive-cut isolated defects surviving every prescribed finite post-arrival horizon. |
| A Rule-30 alternating right boundary obeys the NOR-of-three return and avoids `11` and `00000` | **Accept.** | These are exact local necessary conditions.  They do not characterize the trace language. |
| A uniform charge bound on all `11`-free boundary words | **Accept as sufficient; reject as proved.** | Such a bound would exclude a finite left half, but neither the documents nor the focused verifier supplies the required uniform induction or potential. |

## Triangular interface and the 2-adic obstruction

With lag `d` attached to physical coordinate `cut-time+d`, Rule 30 is
`F(l,c,r)=l XOR (c OR r)`.  Subtracting the ether update gives

```text
delta'_d = delta_d
  XOR (1 XOR e_-) delta_(d-1)
  XOR (1 XOR e_0) delta_(d-2)
  XOR delta_(d-1) delta_(d-2).
```

The coefficient names depend on which of the two lower coordinates is called
`e_0`, but the implemented coordinates and the Boolean identity agree.  The
coefficient of `delta_d` is one and every other term has lower lag.  Therefore
every finite-prefix step, its 28-step return `Q_L`, and every phase prefix map
are unitriangular permutations.  If the first `i` coordinates have period
`P`, the next coordinate is translated by one Boolean cocycle after `P`
returns; its period is preserved or doubled.  This is a uniform induction,
not an extrapolation from the exhaustive checks, and proves that every
`L`-bit `Q_L` orbit has period dividing `2^(L-1)`.

The phase arithmetic is also consistent.  For `p=a+28m`, advancing to the
cut gives `A_a Q_L^m(1)`.  At `a=26`, the formal parameter `m=-1` is the cut
`p=-2`, so its arrival occurs at time `-2`, not at time zero.  The exact
backward states are

```text
time  0: 1
time -1: 1
time -2: 3.
```

At time `-2`, the actual arrival half is phase-two ether XOR `11`.  A direct
Rule-30 step with boundary zero maps it to phase-three ether.  Only output
indices 0, 1, and 2 can see the two changed input bits, and checking those
three closes the boundary-merge issue; a bi-infinite ether continuation is
not being silently imposed.

For the ordinary cuts

```text
p_k = 28*2^k-2,       m_k=2^k-1,
```

one has `m_k=-1 mod 2^(L-1)` whenever `k>=L-1`.  The all-width
unitriangular exponent, together with the exact formal predecessor above,
therefore makes the first `L` arrival bits identical.  An `H`-macrotime
post-arrival test reads at most arrival indices `0,...,2H+1`, so `L=2H+2`
is sufficient and `k>=2H+1` supplies every requested horizon.  This proves
the unbounded horizon symbolically.  The small-width samples in the checker
are diagnostics only and are not the induction.

The physical scope needs two distinct formulations.  On the infinite
periodic ether background, this is an isolated one-bit defect.  For an
ordinary finite half `n_r OR 2^p`, there is also a later truncation defect at
`7r`; choosing `7r>p+2H+1` keeps that second interface outside the complete
dependency cone.  Thus the construction also gives finite one-bit
modifications of ether truncations with unbounded post-arrival delay, but it
does not turn the formal negative cut itself into an ordinary configuration.

The smaller bit-18 example also checks out.  The corrected
`tools/verify_interface_obstructions.py` now reconstructs the dependency cone
from the isolated periodic tail, checks the arrival word and delay, and checks
representative finite truncations while leaving the all-`r` step to the stated
locality inequality.  Direct elementary-rule evolution gives the
low-index-first arrival word

```text
1010100001000001011010
```

and macro residues `1,1,1,2,2,1,1,1,2,1,0`.  Evolving the original finite
halves gives `q(n_r OR 2^18)=19` for every tested `r=6,...,12`; the written
all-`r>=6` conclusion follows from locality because the first truncation
defect is then at index at least 42, beyond the required index 39.

## Fixed-position insertion obstruction

Let `n_R` have ones at `0,7,...,7(R-1)` and delete the one at 42.  The result
agrees through index 43 with the six-one word `n_6`; its next possible one is
at 49.  The first failed latch of `n_6` is macrotime 21, and that test uses
exactly initial indices `0,...,43`.  Hence locality proves
`q(m_R)=21` for every `R>=7`, including both the passing tests and the failing
test at 21.

For odd `R`, the separately established period-seven truncation theorem gives
`q(n_R)=(7R-1)/2`.  Restoring the same absent bit at position 42 therefore
changes `q` by `(7R-43)/2`, which is unbounded.  The result rules out any
finite insertion allowance depending only on the inserted position.  It does
not rule out a charge depending on the retained outer tail, total weight, or
another nonlocal state.

This argument inherits the odd-`R` theorem; the focused obstruction checker
replays `q(n_6)=21` and the cone equality but does not reprove that imported
infinite-family theorem.  The dependency is explicit and legitimate, rather
than concealed finite checking.

## Unrestricted charge counterexamples and physical coherence

Two elementary implementations reproduce

```text
n = 4329177217 = 0x1020a0081,
support = {0,7,17,19,25,32},
wt(n)=6, q(n)=26.
```

The sparse-prefix enumeration is locality-complete at fixed weight: failure
by macrotime 26 reads 54 initial bits, while any of the at most six ones
outside that window cannot change an earlier latch.  Exhausting all surviving
prefixes of weight at most six through the next latch proves `q<=26` for all
ordinary integers of weight at most six, and the displayed word makes the
bound sharp.  This proves the unrestricted sharp-charge inequality false.

It does not produce a physical right half.  The unique free even-time boundary
used by this witness is

```text
01010100110101101101010011,
```

which contains `11` at positions 8, 13, 16, and 24.  Consequently `q=26`
measures necessary left-cone latch length under freely chosen right inputs; it
is not 26 macrotimes of one coherent Rule-30 orbit.  Any use of this witness
against a physically coupled theorem must first supply a coherent right side,
and this witness cannot.

A further, not-yet-artifacted witness supplied for this review also checks
directly:

```text
n = 0x10140001020a0081,
support = {0,7,17,19,25,32,50,52,60},
wt(n)=9, q(n)=40.
```

An elementary shrinking-cone replay passes 40 latches and fails the next;
direct packed iteration first reaches residue 3 at macrotime 40.  Thus it
refutes `q(n)<=4 wt(n)+2` (indeed every integer additive constant at most 3).
This should remain a reviewed observation until a certificate and independent
verifier are added.  Its free boundary also contains `11`, so it provides no
physical-coherence upgrade.

## Alternating right-boundary coherence

For the right half, bit zero is nearest the centre.  One step under prescribed
centre `b` is exactly

```text
R_b(n)=((n<<1)|b) XOR (n|(n>>1)).
```

At even phases the return is `H=R_1 o R_0`.  Exhausting the three relevant
input bits, or simplifying the truth table, gives

```text
s_(k+1) = (1-s_k)(1-u_k)(1-v_k).
```

This NOR-of-three identity immediately forbids `11`.  The stated five-bit
transitions are also correct:

```text
010 -> 000,
001 -> 010,
011 -> 00(d OR e).
```

When the current and next boundary bits are both zero, the remaining low pair
cannot be `00`.  The three cases above reach low triple `000` within at most
three more returns, after which NOR emits one.  Therefore four zeros force a
one and `00000` is forbidden.  The cone enumeration uses exactly `2L-1`
initial right cells for a boundary word of length `L`; it finds 17 length-six
words and excludes `101001`.  This confirms that forbidding `11` and `00000`
is a sound relaxation, not an exact trace-language presentation.

The centre coupling is rule-specific.  For Rule 30, the even `0->1`
transition forces `left=1-right`, and the odd `1->0` transition forces
`left=1`.  For Rule 15, `F=1-left`, so alternation requires `left=centre` and
does not constrain the right neighbour.  For Rule 90, the condition is
`left XOR right=next_centre`.  The focused controls use these scopes
correctly: Rule 15 supplies a genuine alternating control, while the Rule 90
zero-left result passes only the necessary left-cone test and is not asserted
to be a full fibre.

Finally, a constant `C` satisfying

```text
8 W_N-N >= -C
```

for every `N` and every `11`-free free-boundary word would be sufficient.  A
physical alternating orbit supplies such an `11`-free word; a finite initial
left half makes `W_N` eventually constant, contradicting the inequality as
`N` grows.  The logic is correct and even proves a stronger relaxed theorem
than required because the actual trace language is smaller.  No current
artifact proves this uniform inequality.  The sideways state expands with
the cone, and finite enumeration cannot replace a uniform induction or a
potential with controlled endpoints.

## Seven-column phase slips

The formulas in `golden-mean-phase-slip.md` are exact under the declared sole
hypothesis `s_k s_(k+1)=0`.  In particular, evaluating the seven displayed
even-time entries at time zero gives

```text
e_1+...+e_7 = 4-2s_1-s_2-s_3.
```

The two adjacent-pair constraints permit at most three units in the
subtracted expression, so every first slab has weight at least one.  This is
a first-slab fact, not a restartable renewal statement.

For `d_i=1[s_i s_(i+1)=00]`, direct substitution in the depth-seven pair
gives

```text
Delta_k = A_k+B_k-1
        = d_(k+2)-d_(k+1)+d_(k+1)d_(k+2).
```

The product is exactly the indicator of the length-three word `000`.
Summing from `k=0` through `M-1` telescopes with no index shift left over:

```text
sum Delta_k = d_(M+1)-d_1
            + #{i=1,...,M : s_i s_(i+1) s_(i+2)=000}.
```

Thus every temporal prefix at the first cut has excess at least `-1`.  The
55 eight-bit golden-mean words exhaustively checked by the focused verifier
cover the local formulas; the telescoping conclusion itself is symbolic and
holds at every length.

Accept the stated obstruction to transporting that excess by itself.  For
the golden-mean word `s=0^infinity`, sideways reconstruction gives the
spatially alternating initial left half `101010...`, with every temporal
column constant.  Column seven is constant one and column fourteen is
constant zero, so the second cut has excess `-M` over `M` temporal pairs.
This word violates the physical `00000` restriction, but the proposed lemma
quantifies over the larger `11`-free shift and must handle it.  Consequently
the coboundary in the first cut is not, by itself, a spatially propagated
charge.

The physical finite witness also checks out at its claimed useful horizon.
For left integer `857638` and right integer `17`, a direct full-row Rule-30
evolution gives centre trace

```text
01010101010101010101010101010
```

through times `0,...,28`.  Its even-time right-neighbour word through time 28
is `101000010010010`, which has no `11`, and initial-left indices `21,...,27`
are all zero.  This refutes an aligned seven-cell renewal which forgets the
incoming cut state even on a genuinely coherent finite spacetime prefix.  It
does not refute a global charge inequality: the preceding slabs may carry the
credit consumed by the zero slab, and the witness asserts no infinite
alternating orbit.

One sentence in the phase-slip note exceeds the focused evidence: it says a
left-permutive reconstruction was carried through depth 112 and checked
against 113 centre bits.  The verifier and its tests check the displayed
truncated row only through time 28, and no saved depth-112 artifact or
reproducer is present.  The time-28 fact is sufficient for the aligned-block
obstruction.  The depth-112/113 assertion should be removed, reduced to the
verified horizon, or supplied with its missing reproducer.

The weight-cap table is correct but likewise absent from the focused
phase-slip verifier.  An independent layered enumeration, adding exactly two
initial cells per latch and discarding a prefix as soon as either its Rule-30
latch fails or its forced even-time boundary forms `11`, reproduces maximum
joint depths `40,43,45,49` for caps `11,12,13,14`, with the next layer empty.
Locality makes this exhaustive over unrestricted support at each fixed cap.
It remains bounded evidence, not an induction in weight.

Lead resolution: the unnecessary depth-112/113 sentence was removed; the
note now asserts exactly the independently checked time-28 witness. The
weight-cap table is reproducible with
`experiments/alternating_right_coherence.py --weight WEIGHT --bound DEPTH`;
the lead independently replayed the four rows through weight 14.

The proposed potential inequality has the correct direction.  If every
seven-column slab satisfies

```text
8 B-7 >= Phi(next)-Phi(current),
```

then summing gives total charge at least `Phi(final)-Phi(initial)`.  A uniform
lower bound on `Phi` over all reachable cuts and a uniform upper bound on
`Phi` over all permitted initial cuts therefore give the desired constant.
The residual zero through six initial cells decrease the bound by at most six.
Both endpoint signs are essential.

No such `Phi` has been constructed.  Its cut state must retain exact adjacent
temporal columns, their shared predecessor data, and their lineage from the
centre boundary; otherwise consecutive locally valid slabs need not compose.
Identity (5) is a possible component, while the constant-column example proves
it cannot be the whole functional.  Finite truth tables for the first slab,
the physical reset witness, and the weight-cap enumeration do not establish
the spatial transport inequality or either uniform endpoint bound.

## Next implication

The composition route must retain nonlocal wake or tail credit: neither a
fixed per-interface delay nor a fixed marginal-bit allowance survives.  The
right-coherence restriction removes the known weight-six and weight-nine
free-boundary counterexamples from the physical problem, leaving a precise
proof target: construct the exact-lineage cut functional above, or otherwise
establish a uniform lower bound `8W_N-N>=-C` on the golden-mean input shift or
the smaller exact right-trace language.  Temporal excess at one cut and
state-free aligned-block renewal are both now exactly obstructed.  Until the
spatial transport and endpoint bounds are proved, the interface results are
obstructions and reductions, not an exclusion of eventual period two.

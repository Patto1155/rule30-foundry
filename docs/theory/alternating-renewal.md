# Fixed-weight renewal: exact scope and a block counterexample

This note isolates a possible induction target. It does not establish the
cone-charge lemma or exclude period two. No prize-status grade is changed.

The subsequent weight-six counterexample has `q=26>4*6`; the unrestricted
sharp charge target is therefore false, although the fixed-weight certificates
below remain valid. See `../experiment-logs/2026-09-19-zero-tail-composition.md`
for the exact obstruction and the weaker physical-boundary target.

Let `q(n)` be the first `k >= 0` for which `G^k(n) mod 4` is neither 1 nor 2.
Passing `m` latches means `q(n) >= m`, with infinite `q` allowed provisionally.
A passed left-only test never asserts right-half compatibility.

## Locality and fixed-weight certificates

The test at macrotime `k` depends only on initial bit indices `0..2k+1`.
Indeed each Rule 30 time step enlarges dependence toward the deeper left by
one cell, and the residue test initially uses two cells. Consequently, proving
that no prefix of length `2(B+1)` and weight at most `r` passes `B+1` latches
certifies `q(n) <= B` for every ordinary integer of weight at most `r`, with
**no restriction on support width**. Higher initial ones cannot rescue a
failure inside the tested cone.

The existing finite minima through depth 44 already imply:

| Weight cap `r` | Sharp bound `B_r` | One witness `n` |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 3 | 1 |
| 2 | 8 | 129 |
| 3 | 11 | 4225 |
| 4 | 16 | 1065089 |
| 5 | 19 | 1075855489 |

These are fixed-weight finite certificates, not a theorem for arbitrary `r`.
The independent checker `experiments/alternating_renewal.py` reproduces the
bounds using the elementary rule truth table. It adds two initial cells at a
time, prunes excessive weight, and retains exactly prefixes passing the
necessary centre constraints. It does not call either the integer map or the
sideways reconstruction. For `r=5`, no layer contains more than 26 retained
prefixes; the layer at 20 latches is empty. This independently rechecks existing
horizons rather than extending them.

```bash
python experiments/alternating_renewal.py --weight 5 --bound 19
python -m unittest tests.test_alternating_renewal
```

The same checker keeps the zero initial word alive for Rule 15, as required
by its actual finite alternating fibre. It also keeps that word alive under
Rule 90's necessary left-only constraints: either right-neighbour value can
supply the desired next centre bit. The Rule 90 result is not a claim of an
actual alternating fibre.

## A precise induction reduction, and its missing step

Suppose a bound `B_(r-1)` has been proved for weight at most `r-1`. Let `n`
have `r` ones, highest one at bit index `p`, and let `n' = n - 2^p`.
If `p > 2 B_(r-1) + 1`, locality gives

```text
q(n) = q(n') <= B_(r-1).
```

Thus an induction step only needs to handle `r`-one integers below
`2^(2 B_(r-1)+2)`. This class is finite, but its finiteness does **not** prove
its orbits terminate. An unbounded iteration loop is not a proof certificate.
A usable next lemma would provide a uniform, Rule-30-specific termination
argument for this remaining class, or a charge preserved by adding the last
one. No such argument is supplied here.

Existence of some `B_r` for each `r` is equivalent to no finite integer
surviving forever, not an independent compactness shortcut. For the reverse
implication, the set of infinite binary words having at most `r` ones is
compact: its complement is witnessed by a finite prefix with `r+1` ones.
If arbitrarily long survivor prefixes of weight at most `r` existed, nested
compact survivor sets would have an infinite survivor of weight at most `r`,
which is an ordinary integer. This proves the equivalence only. It also
explains why generic compactness cannot finish the problem; Rule 15 provides
a real survivor in the analogous setting.

## A block renewal rule fails exactly

The assertion “every aligned block of eight initial cells contains a one”
is false. Take `n=129`. Its first 16 cells, nearest the centre first, are

```text
10000001 00000000.
```

It passes eight latches and fails the ninth (`q(129)=8`). Sideways construction
also obtains it from the free boundary word `01010100` (integer 42). The second
block has zero weight, while the first block has weight two. For charge
`8*weight - length`, their charges are respectively `+8` and `-8`; their sum
is zero. This is an exact left-cone witness, not a full-orbit assertion, and it
does not refute the total cone-charge bound.

There is a stronger obstruction to a naive one-one induction. No universal
transformation can remove one initial one and lose at most four passed latch
events, even if arbitrary rearrangement is permitted: it would transform
`n=129`, which passes eight, into a weight-at-most-one word passing at least
four. The sharp fixed-weight-one bound is three. Therefore a per-one step
must sometimes lose at least five latches or retain a nontrivial credit state.

Pairing two ones does not make plain spatial deletion work. The sharp
weight-five witness `1075855489` has ones at zero-based positions
`0,7,14,21,30` and passes 19 latches. Delete any contiguous 16-bit block
containing exactly two ones and close the gap. The only resulting integers
are `16417` (failure at 2) and `16513` (failure at 10). Neither passes the
required `19-8=11` latches. A paired-one/eight-macrostep reduction therefore
needs a boundary repair or a changed boundary condition.

Therefore removing the first block and resetting the cut state discards
necessary credit. Any eight-cell discharge must permit the first block to
carry enough potential difference to pay for the second block. A concrete next proof obligation is to construct a finite set of
boundary-repair rules for a 16-cell deletion, reducing weight by two while
losing at most eight latches, with a separately handled finite base class.
The rules must specify an exact repair radius, retained cut state, and allowed
replacements. Each rule must preserve the remaining cone constraints; merely
claiming that a lower-weight survivor exists repeats the desired bound.
Alternatively, the tile track can retain explicit credit. Retaining an
unbounded total weight count alone restates the target; useful credit needs
local transition semantics and an independently proved bound.

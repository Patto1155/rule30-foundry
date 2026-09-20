# Second-slab phase-slip identities and their transport ceiling

This note extends `golden-mean-phase-slip.md`. It does **not** construct a
spatial charge functional or exclude an alternating centre. The exact identities
below follow from finite Rule-30 truth tables and hold uniformly wherever their
stated word hypotheses hold.

## The seven initial cells in the second slab

Write `(a,b,c,d,e,f)=(s_1,...,s_6)` for a word avoiding `11`. Sideways
reconstruction gives the second slab's initial weight

```text
W_[8,14] = 3 - 2c + 4ac - bd - 2e + 3be + 6ce
             - 8ace + 2cf - 3acf + df.                 (1)
```

In particular `W_[8,14]>=1`, with a simple exhaustive symbolic proof:

| Conditions | Weight |
|---|---|
| `c=1, a=0` | `1+4e+2f` |
| `c=1, a=1` | `1+3(1-e)+(1-e-f)` |
| `c=0, e=1` | `1+3b` |
| `c=0, e=0` | `2+(1-bd)+df` |

Every displayed summand after the constant is nonnegative under `11` avoidance.
This is a two-slab identity, **not** an induction: no arbitrary incoming cut is
reset to the original alternating column. Its negative `ace` term shows one
specific new interaction absent from the first-slab linear weight formula.

## Exact second-slab coboundary on the gap-three/five relaxation

Consider the one-sided right-boundary language whose words avoid
`11`, `101`, `10001`, and `00000`. Its successive ones are separated by
three or five macrosteps, and its initial zero run has length at most four.
The initial-run condition matters: the gap condition alone would admit an
arbitrarily long initial zero run or the all-zero word. An independent right-coherence calculation
identifies a robust right-half invariant whose output belongs to this language
(see `right-coherence-followup.md`);
the identity here needs only the language assumption. No claim that *every*
physical right trace satisfies it, or that every such word is physically
realizable, is used.

Let `A14(k),B14(k)` be the even/odd cells at depth fourteen, and put
`Delta14(k)=A14(k)+B14(k)-1`. Define `F` on admissible five-letter words by

```text
F(01000)=F(10000)=F(00010)=1,
F(00100)=F(01001)=F(10010)=F(00001)=0.
```

There is an exact identity

```text
Delta14(k) = 1[s_(k+2)..s_(k+7) = 100001]
             + F(s_(k+3)..s_(k+7))
             - F(s_(k+2)..s_(k+6)).                 (2)
```

For completeness, the intermediate Boolean-polynomial simplification is

```text
Delta14(k) = -1 + s_(k+3) + 2s_(k+4) + 2s_(k+6) + s_(k+7)
             + s_(k+2)s_(k+5) - 2s_(k+3)s_(k+6)
             - 2s_(k+4)s_(k+7).
```

Before simplification, the sideways radius bounds put the two depth-fourteen
cells inside `s_k,...,s_(k+7)`: each two spatial reconstructions advances the
maximum temporal input by at most one macrostep. The displayed simplification
reduces the dependency to the six letters in (2). Thus the finite truth table
exhausts every possible local context of an *infinite* renewal word, rather than
sampling temporal-periodic trajectories. Its eight directed edges between
five-letter contexts also retain their actual overlaps; no predecessor is
chosen afresh at a temporal step.

The complete local check has only eight rows:

| `s_(k+2)..s_(k+7)` | `Delta14(k)` | Production |
|---|---:|---:|
| `000010` | 1 | 0 |
| `000100` | -1 | 0 |
| `001000` | 1 | 0 |
| `001001` | 0 | 0 |
| `010000` | 0 | 0 |
| `010010` | 0 | 0 |
| `100001` | 0 | 1 |
| `100100` | 0 | 0 |

Summing (2), for every `M`, gives

```text
sum_(0<=k<M) Delta14(k)
  = #{k<M: s_(k+2)..s_(k+7)=100001}
    + F(s_(M+2)..s_(M+6)) - F(s_2..s_6) >= -1.    (3)
```

Thus the first-slab nonnegative-production mechanism has a genuine second-slab
analogue on this stronger language: production counts five-step gaps. The
first-slab `000` count produces twice per such gap; their production rates are
different, so these are not interchangeable potentials.

## The third slab still requires initial-row accounting

The language condition does not solve the transport problem. Take the periodic
word `s=(10000)^infinity`. At depth twenty-one, the sum of `A21+B21-1` over its
five temporal phases is `-2`. Consequently its temporal prefix excess is
unbounded below. No bounded temporal coboundary with nonnegative production can
represent that third-cut excess. This obstruction holds within the gap-three/
five relaxation itself; physical realizability of this entire infinite word is
not asserted or needed to obstruct a proof on that relaxation.

Nor does the stronger language permit a state-free block renewal. The prefix

```text
0010000100100100001000010
```

avoids all four forbidden words and reconstructs seven zero initial cells at
depths `22..28`. It has an infinite gap-three/five extension. Thus the earlier
empty-block obstruction persists after this refinement.

The resulting reduced obligation remains spatial: incorporate the emitted
initial-row weight and the full inherited cut into a functional `Phi` satisfying

```text
A * slab_initial_weight - 7 >= Phi(next cut)-Phi(current cut),
```

for some finite `A`, with a uniform lower bound on reachable terminal cuts and a
uniform upper bound on initial cuts. Equations (2)--(3) provide an exact new
component and a small endpoint table; the depth-twenty-one obstruction explains
why using this component alone cannot complete the proof.

## Reproduction

Run `python experiments/phase_slip_transport.py`. It checks (1) on all 89
nine-letter golden-mean words, checks (2) on all 14 nine-letter gap-three/five
words, and verifies the third-cut obstruction with a separate periodic-time
truth-table recurrence. Finite-word checks verify local Boolean identities;
the uniform telescoping statement follows from summing (2), not from a survival
scan. None of these calculations claims a formal proof-assistant proof.

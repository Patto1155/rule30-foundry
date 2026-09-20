# Zero-tail composition: exact obstructions and physical phase-slip charge

Date: 2026-09-19. Continues local commit `520d39a` on
`research/slanted-cone-charge`, itself continuing `ca48894`.
No push, merge, prize-grade change, or JEV request.

## Goal and setup

Test whether the proved period-seven truncation interface can compose across
additional initial ones. Reuse the left transfer, sparse-prefix renewal,
sideways reconstruction, and independent elementary-rule cone tools. Keep
the distinction between a necessary left-only survivor and a full orbit.

The route changed when a targeted six-one test refuted the proposed sharp
unrestricted charge. The replacement sufficient lemma retains physical
right-boundary coherence rather than trying to exclude a larger, unphysical
class at the same sharp constant.

## Exact results

### The unrestricted sharp charge is false

For `G=T_1 o T_0`, let `q(n)` be the first failed residue test. Then

| Initial left integer | One positions, zero based | Weight | `q` | `8W_(2q)-2q` |
|---|---|---:|---:|---:|
| `4329177217` | `0,7,17,19,25,32` | 6 | 26 | -4 |
| `1158551008470237313` | `0,7,17,19,25,32,50,52,60` | 9 | 40 | -8 |

The first row already violates `8W_N-N>=-1` at `N=50`, where its charge is
`-2`. Its highest-one parent has `q=16`; adding bit 32 increases lifetime by
ten. Exact sparse-prefix enumeration also proves the sharp all-support-width,
fixed-weight theorem `wt(n)<=6 => q(n)<=26`. This is not an induction in weight.

The second row falsifies the named repair `q(n)<=4 wt(n)+2`. It contains the
first row and adds bits 50, 52, and 60. Attempting to repeat that rescue motif
does not produce an unbounded-deficit family. Thus a finite deficit is still
possible; these examples require `C>=8` in `8W_N-N>=-C`.

Both rows have a forced even-right word containing `11` at indices 8–9.
They therefore do **not** realize a full alternating centre for all their
left-only latch times. The counterexamples refute the stronger free-boundary
charge assertion, not finite-configuration period-two exclusion.

The weight-six certificate is
`runs/theory/alternating-charge-counterexample.json`, with a standalone
truth-table verifier. The weight-nine row was separately replayed through
40 successful latches and failure at 40 using that verifier's cone evaluator.

### Neither fixed insertion credit nor fixed interface delay composes

Let `n_R=sum_(j<R)2^(7j)`. For every `R>=7`,

```text
q(n_R - 2^42) = 21.
```

This follows from the exact `n_6` residue word and locality through bit 43.
Restoring the fixed bit 42 gives, for odd `R`,
`q(n_R)=(7R-1)/2`. The marginal lifetime gain is unbounded. A single insertion
can make arbitrarily many remote, already-present ones relevant.

This is not only a left-only obstruction. Give the full finite row right half
`v_R=sum_(j<R)2^(5+7j)` and centre zero. The restored row agrees with the
period-seven ether until its nearest missing left one reaches the centre;
its exact first centre mismatch is at time `7R+1`. Deleting left bit 42 makes
the exact first mismatch time 43 for every `R>=7`. Causality and the unique
extreme left input prove both assertions; independent full-row truth-table
replays check the coordinate conventions. Thus even physical finite rows
have an unbounded marginal insertion gain, with the same right half in each
pair.

There is a stronger obstruction to a constant allowance after the first
discrepancy arrives. The period-28 moving-interface return on `L` bits is
unitriangular, fixes bit zero, and has order dividing `2^(L-1)`. The formal
2-adic parameter `p=-2` has arrival defect `11`, and

```text
T_0(E_phase2 XOR 3) = E_phase3.
```

It merges exactly into the eternal ether. Ordinary cuts
`p_k=28*2^k-2` reproduce any prescribed finite prefix of this arrival state.
For any `H`, choose `k>=2H+1` and `7r>p_k+2H+1`. Then the ordinary finite
configuration `n_r OR 2^p_k` passes every latch through `H` macrosteps after
arrival. There is no uniform post-arrival horizon even for one isolated
added bit. This is an all-parameter proof, not an enlarged finite scan.

The handwritten induction, phase alignment, inverse states, boundary merge,
and locality bridge are independently checked in
`tools/verify_interface_obstructions.py`. Software checks the finite Boolean
identities supporting the induction; it is not a formal proof-assistant proof.
See `../theory/triangular-interface-cocycle.md` and
`../theory/one-bit-delay-obstruction.md`.

### Physical right coherence supplies a uniform decreasing rank

For the forced centre `01`, the right-half return is

```text
R_b(x)=((x<<1)|b) XOR (x|(x>>1)),  H=R_1 o R_0.
s_(k+1) = (1-s_k)(1-u_k)(1-v_k).
```

Thus `11` is forbidden. On the zero-output low triples encoded by
`0,2,4,6`, ranks `0,1,2,3` strictly decrease at every zero-to-zero return.
The longest chain is `6->4->2->0->odd`, so `00000` is also forbidden and
every five observations contain a one. The graph is a necessary factor,
not a complete symbolic presentation: it admits `101001`, which the exact
cone rejects.

This gives the weaker sufficient target: prove one uniform finite-deficit
charge for sideways words avoiding `11`. Every actual alternating orbit
supplies such a word. At any eventual-alternation onset, restart at a
centre-zero phase; the left half still has finite weight, contradicting a
uniform charge. No exact characterization of right traces is needed.

The no-`11` sparse-prefix check has exact maximum jointly valid depths

```text
weight cap:  1  2  3  4  5  6  7  8  9 10 11 12 13 14
depth ell:  3  8 10 14 17 21 24 28 31 39 40 43 45 49
```

These are fixed-weight finite certificates. `ell` also checks the inferred
right word and is not `q`: for example the four-one ether truncation has
`q=16` but `ell=14`. The named bounded falsifier of `ell<=4 wt` finds none
through weight 14. This is not evidence of a completed induction.

### A checked phase-slip identity, and the precise transport gap

For `11`-free `s`, define `d_i=1[s_i s_(i+1)=00]`. Let `(A_k,B_k)` be the
even/odd pair at spatial depth seven in sideways reconstruction. Then

```text
Delta_k = A_k+B_k-1
        = d_(k+2)-d_(k+1)+d_(k+1)d_(k+2).
```

The last term is the `000` indicator. Summing gives an exact boundary term
plus nonnegative production, so every temporal prefix has excess at least
`-1`. This is a checked uniform identity on the first seven-column slab;
it is not the initial-row cone-charge inequality.

The first cut cannot be reset to alternation. Even exact physical right
coherence admits an empty aligned seven-cell initial-left block: left integer
`857638`, right integer `17`, and centre zero give alternating centre bits
through time 28, but initial-left indices `21..27` are all zero. This is
independently replayed as a full finite row. Also, on the golden-mean word
`s=000...`, temporal column seven is all ones and column fourteen all zeros;
the temporal excess alone cannot be propagated monotonically.

The single best next obligation is to transport the explicit boundary term
and production through a second seven-column slab **with its initial-row
weight included**, retaining shared predecessors. A sufficient certificate is

```text
8 * slab_initial_weight - 7 >= Phi(next cut) - Phi(current cut),
```

with a uniform lower bound at all exact cuts and a uniform upper bound at all
initial cuts. The first-slab identity supplies a candidate component, not a
complete `Phi`. See `../theory/golden-mean-phase-slip.md`.

## Reproduction and verification

```bash
python experiments/alternating_charge_counterexample.py --verify runs/theory/alternating-charge-counterexample.json
python tools/verify_alternating_charge_counterexample.py runs/theory/alternating-charge-counterexample.json
python tools/verify_interface_obstructions.py
python experiments/alternating_right_coherence.py
python experiments/alternating_right_coherence.py --weight 14 --bound 49
python tools/verify_golden_mean_phase_slip.py
```

Rule 15 and Rule 90 retain their distinct centre equations and negative-control
scopes. They are not made to obey the Rule-30-specific right return map.
The baseline repository gate passed with eight passes and three expected
skips: two absent canonical bitstreams and the absent SAT/DRAT toolchain.
The final serialized `python tools/verify_all.py` likewise passed: eight
passes, three expected skips, zero failures. The independent weight-six,
interface-obstruction, right-coherence, and phase-slip replays all passed.
The four no-`11` fixed-weight rows 11–14 were independently reproduced by the
lead and the hostile reviewer. `git diff --check` passed. Claim-ledger grade
cells were compared with the starting commit and are unchanged; only the
refuted uniform extension and next obligation were corrected.

## Delegation and acceptance

Ten GPT-5.6 Sol agents at medium reasoning were used, with two to four active
during the research waves. Each had narrow ownership; proof synthesis and
acceptance remained with the Astra lead. Agreement was not counted as evidence.

| Agent | Narrow obligation |
|---|---|
| composition_derivation | Last-one and fixed-hole composition argument |
| charge_falsifier | Targeted sharp-charge falsification and producer |
| triangular_interface | Unitriangular cocycle and inverse-time obstruction |
| charge_independent_verifier | Independent cone enumeration and certificate audit |
| deficit_recurrence | Named finite-deficit repair and repetition falsifiers |
| right_boundary_coherence | Physical right return, rank, and coherent enumeration |
| interface_obstruction_verifier | Independent all-parameter interface proof audit |
| right_trace_language | Independent right-factor/rank derivation |
| coherent_charge_derivation | Seven-column formulas and phase-slip proof obligation |
| composition_hostile_review | Hostile scope, phase, locality, and proof audit |

## Interpretation

Period two remains unexcluded. The advance is an exact obstruction to the
old composition programme and a smaller, physically justified proof target
with a nontrivial first-slab charge identity. The all-weight transport lemma
is still missing. No finite scan or agent consensus is substituted for it.

## Breakthrough assessment and committed theories

The work and proof notes above were committed in `f7812ab`; the preceding
period-seven truncation theorem and predecessor-coherence results are in
`520d39a`. There was **no breakthrough establishing general period-two
exclusion**. There were substantive, rigorously checked advances within the
research programme:

| Result | Assessment |
|---|---|
| Every period-seven truncation `n_r` has `q(n_r)<=4r` | An all-parameter theorem for a restricted family, proved in the preceding session. It does not cover arbitrary rows or later alternating onsets. |
| The sharp unrestricted charge is false | A decisive correction to the proposed sufficient lemma, witnessed by six initial ones with `q=26`. A uniform finite deficit is not ruled out. |
| No constant marginal insertion or isolated post-arrival allowance works | Uniform impossibility results for these composition strategies, not just failures at a chosen scan horizon. The insertion obstruction also holds for actual full finite rows. |
| Physical right-boundary rank and seven-column phase-slip identity | A positive structural mechanism: the rank forces bounded zero runs, and the identity separates boundary debt from nonnegative production. It has not been transported through arbitrary spatial slabs. |

The remaining theory is explicitly a conjectural certificate, not a proved
charge bound. It seeks a functional on exact cuts satisfying
`8*slab_initial_weight-7 >= Phi(next)-Phi(current)`, with a uniform lower
bound at reachable cuts and a uniform upper bound at initial cuts. The
committed notes give the first-slab identity, exact obstructions to resetting
the cut, and the missing second-slab transport obligation.

The breakthrough assessment therefore supports continuing this narrower
mechanism, while rejecting any announcement of a solution to period two or
any change to the prize grades. The next mathematical action is the transport
proof, rather than a larger survival scan.

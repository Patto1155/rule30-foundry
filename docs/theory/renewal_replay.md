# Alternating renewal: recovered certificate replay

This is a source audit and exact replay, not a period-two exclusion. Ledger
admission and grades remain the lead's responsibility.

## Provenance and acceptance

Recovered source: `origin/research/slanted-cone-charge` at
`44d11eebaab4e8486c195dee1e5c61a36860851f`. Seven original experiment modules
were restored byte-for-byte, including the physical-cut certificate's 45
integer-weighted rows. No branch merge or original state-file restoration was
performed. `renewal_replay.py` is a new small acceptance wrapper.

```bash
python experiments/counting_bound.py --pretty
python experiments/renewal_replay.py --out runs/renewal-replay-2026-10-14/results.json
```

The counting tool bounds DFAO behavior classes, not sparse Rule-30 cones.
Here the declared purpose is **exact-exclusion**, never complexity evidence.
For weight cap r and initial cone length n, the tested class has
`sum(j=0..r) binomial(n,j)` words; the r=5 certificate uses n=40, giving760099 classes and
log2(class)=19.535827810765824 against40 subsequent target centre bits.
The wrapper computes each actual cap's count **before** replaying the empty
layer. A random-word complexity interpretation is deliberately rejected;
the left-only cone additionally permits free right inputs, so the DFAO
counting comparison is not a statistical model of that relaxation. Every longer
left integer projects to one of these words. Independent local truth-table
pruning gives an empty final layer; no probabilistic inference is needed.
The graph certificates quantify over every exterior bit pair. The transport
replay checks an existing positive integer contradiction, and runs no new
LP infeasibility search.

## Claim classification

“Reproduced” means the exact finite claim was rerun, not Lean checked.
“Proved” below identifies an analytic implication with its premises stated.

| Source / substantial claim | Classification | Exact replay or qualification |
|---|---|---|
| alternating-renewal: fixed weight bounds for caps 0..5 | Reproduced | Sharp lifetimes 0,3,8,11,16,19; all-support-width by cone locality. |
| alternating-renewal: every aligned block of eight contains a one | Wrong | 129 has initial blocks `10000001 00000000`, and lifetime 8. |
| alternating-renewal: remove one one losing at most four latches | Wrong | Weight-two lifetime 8 cannot reduce to weight-one lifetime >=4; sharp weight-one bound is 3. |
| alternating-renewal: plain 16-cell deletion of two ones loses at most eight latches | Wrong | Weight-five witness 1075855489 has lifetime19; its stated deletions 16417,16513 have lifetimes2,10. No repair rule is disproved by this. |
| alternating-renewal: locality reduction and compactness equivalence | Proved (argument) | Last-one induction narrows a finite class; it does not prove its termination. Compactness equates all fixed-weight termination with absence of a finite left-only survivor. |
| unrestricted sharp charge q<=4 wt(n), and repair q<=4 wt(n)+2 | Wrong | Independently replayed n=4329177217, wt6, q26; n=1158551008470237313, wt9, q40. Their charges at 2q are -4,-8. |
| alternating-right-boundary-coherence: two-step NOR reset | Reproduced | All32 five-bit dependency cones checked against the elementary truth table. |
| right boundary avoids11 and00000; each five outputs includes one | Proved and reproduced | Triple-factor zero ranks descend; no five consecutive zeros; density >=floor(M/5). This is right output density, not initial-left density. |
| 11/00000 avoidance characterizes physical boundary language | Wrong | Exactly17 six-letter words occur;101001 avoids both blocks but is absent. |
| weight-six left charge counterexample is a physical alternating orbit | Wrong | Inferred free boundary contains11 at8,13,16,24, prohibited by physical NOR reset. |
| golden-mean (11-free) uniform left charge >=-C | Reported only / open | The source explicitly proposes it without proof. No new uniform conclusion follows from this replay. |
| right-coherence-followup: closed K and entry cylinders | Reproduced | Width16; K13526 states,59 deletion rounds. Low8 triggers167,168,170 have768 extensions; forward closure1452 states and5808 edges. Every future one gap in K is3 or5. |
| right-invariant-entry: universal basin | Reproduced | Basin21457; rank<=12, strictly descending outside K on all edges. |
| right-invariant-entry: cyclic complement components and outputs | Reproduced | Sizes9,29,2445; output quotients4,16,2092; small periods0001 and0000100001010001, up to phase. |
| small recurrent components are physically persistent | Wrong | Width22 lift of9 has576 states,2304 cones,longest path9; width18 lift of29 has116 states,464 cones,longest path15. Both are DAGs. |
| every actual trajectory enters K or stays in Q eventually | Proved from reproduced certificates | Q has2445 states. If K is never entered, eventual Q residence starts by macrotime100. This is not a K-entry bound: finite right0 enters K at147. |
| every graph path in Q has periodic output | Wrong | Same-base loops of lengths2 and37 yield different74-letter blocks; nonperiodic selectors construct nonperiodic graph paths. No physical predecessor extension is supplied. |
| weighted-renewal-transport: no named width-three nonpositive credit | Reproduced |45 physical finite rows; endpoint balance exact;41 positive histogram coordinates; M=6110652545858014158278360985185689401228 and demand7M. All seven-cell slabs have zero initial weight. |
| weighted-renewal-transport: signed width-two density with width-two local terminal flux also fails | Proved (telescoping argument) | The flux inequality converts it to the refuted width-three nonpositive family plus endpoint terms. No claim about arbitrary signed densities. |
| reported later740-state renewal core D, exceptional3106/3716 components, terminal-clearing theorem | Reported only | Not replayable from fetched sources; do not identify these with K or Q. |

## Physical information retained and lost

The graph E(v,a)=H(v+2^16 a) mod2^16 contains every physical right path but
allows independently chosen exterior pairs at each return. Its closure gives
sound invariants; graph recurrence alone gives no finite-right recurrent
orbit. Lifted small-component certificates retain the extra cells across
time and eliminate spurious recurrence. They still leave Q unresolved.

The weighted-credit rows are full finite physical alternating-prefix rows,
independently evolved without a prescribed centre. Their endpoint tables
balance while their entire ordered cuts do not. The circulation therefore
refutes the specified functional for every coefficient A but cannot be
concatenated into one infinite physical orbit. Its exact scope is bottom2 /
top8 endpoint bits on both diagonals and a phase-labelled nonpositive density
on at most3 adjacent pairs. The signed-width2 enlargement is excluded only
when its terminal bound has the stated local flux. An order-sensitive,
cone-conditioned functional remains a separate target.

The support-growth worker's slow row r=32 was separately classified for64
macrosteps. Every low16 state is in Q and outside K; output is `01` repeated32
times. Each next low16 state was checked from its18-cell elementary-rule
cone. The first14 states are32,153,544,2329,8352,39577,10784,10521,8352,
39577,10784,43289,41120,39577. This licenses no assertion about all future
returns, but warns that K renewal assumptions cannot be applied to this
observed slow case.

The useful recombination is conditional: the physical NOR reset restricts
right words; K licenses gap3/5 renewal; Q needs a separate finite-support
argument. Neither the NOR output density nor these graph ranks constrain
forced initial-left weight without a new ordered bridge. The replay supplies
no such bridge.

## Filters

**Rule90:** the right restrictions and invariant use Rule30's OR in
`R_b`, including NOR reset. The left-only checker preserves zero for12
Rule90 latches (and Rule15), so it does not falsely infer period exclusion
from left permutivity. This control is a necessary-cone control, not an
actual alternating Rule90 row.

**Zero:** forced centre01 is nonconstant, so the zero full row is outside the
hypothesis. A zero left half already fails Rule30's first latch. The driven
right half r=0 nevertheless has a nonzero forced trajectory and reaches K
at147; it must not be confused with the zero full Rule30 orbit. No argument
here treats all initial conditions identically.

## Missing source search and not run

Searched all fetched origin refs under docs/theory, experiments and
handover for740 and terminal-clearing, and the all-ref commit subjects.
Only the recovery account and main's audit mention the later740 component;
other740 matches are unrelated counts or digits in certificate integers.
Recovery source is
`79b72e0a0ad7fc8c38058368b4d658ddc4268254:docs/handover/archive/2026-10-12-recovery-and-continuation.md`.
Reported local commits22ab58f7,d7eecd17,7dd93382,613b4011 are absent from
this fetched object database. This is source unavailability, not a
mathematical refutation of those reports.

Not run: optimizer/discovery modes, new fixed-weight cap6 certificate,
uniform golden-mean charge proof,740-core replay,all-range additive-rank
extension,terminal-clearing replay,Lean,GPU,independent-model review,full
verifier on the final integrated branch. This worker's full verifier passed747
tests with10 test skips and no failures. Its initial missing10M/SAT
stages were then rerun and passed after restoring canonical10M and borrowing
lead SAT binaries. Only the46M bitstream stage remains unavailable. A focused verbose rerun
after restoration leaves6 named unittest skips: three missing46M subtests
and three GPU/cupy checks; all CPU and worker-pool checks ran. Raw output
and remediation checks are in `queue/results/period2-renewal-verifier.txt`.
The wrapper is the targeted acceptance command.

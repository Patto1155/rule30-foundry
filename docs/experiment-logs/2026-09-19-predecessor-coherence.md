# 2026-09-19 — Predecessor coherence and a periodic-ether interface

## Goal and outcome

Continue locally from `research/slanted-cone-charge`, `ca48894`, seeking a
uniform exclusion of finite left halves with an alternating centre. No general
period-two exclusion or uniform cone-charge bound was obtained. The session
produced an exact obstruction to the proposed density architecture and a
uniform escape mechanism for a restricted infinite family of finite halves.
Prize grades are unchanged. No push, merge, or JEV request was made.

Ten GPT-5.6 Sol agents, explicitly at medium reasoning, handled narrowly
owned tasks, with at most four active concurrently. The lead retained proof
acceptance. Agreement among agents is not evidence; the evidence below is
algebra, closed finite dynamics, and independently reconstructed integer checks.

## Shared predecessors: repair and exact limit

The seven-bit two-update relation has 20 adjacent-row states and 44 edges.
Composing independently witnessed single updates instead gives 48 walks. The
four eliminated walks include `000 -> 100 -> 111`.

Overlapping coherent windows remain exact through three updates (100 paths),
but at four updates admit 224 paths versus 208 actual predecessor traces.
The first false path is

```text
000 -> 000 -> 100 -> 011 -> 010.
```

Its first two updates force the seven-bit original `0000001`; the requested
last first component is zero, whereas every shared continuation forces one.
The experiment stops at this first counterexample. Rule 15 and Rule 90 use
their own truth tables, with no transferred Rule 30 latch condition.

## A stronger obstruction than another failed memory size

The coherent pair-density language is infeasible both with and without time
parity, for coefficient eight and for a variable coefficient. Exact integer
identities are `0 >= 2` (nine rows) and `0 >= 6` (24 rows). They use only
zero-emission bottom rows and no initialization or endpoint-bound rows, so
neither a different coefficient nor any finite endpoint deficit repairs them.

More strongly, genuine cuts of the `n=129` latch cone at depths 4 through 7
force

```text
Phi_5 + Phi_7 - 2 Phi_3 >= 6.
```

For a density on one diagonal pair, the same expression is a sum of three
terminal-flow cycles and is at most zero. Those cycles lift to infinite
Rule 30 strips. Thus **enlarging spatial flux memory alone cannot repair this
pair-density/terminal-flow architecture**, even if the entire strip is kept.
This is not a counterexample to cone charge. The exact algebra and scope are
in `docs/theory/coherent-density-obstruction.md`.

One justified repair let the density and bottom correction see two successive
pair rows. Its four variants also have exact contradictions, all `0 >= 4`.
The new obstruction is different: two bulk strips force `e_1=e_2=0`, violating
the first centre latch despite satisfying the final cap. Its terminal cycle
requests `00,00,00,11,00,00`, while shared predecessors force the last pair to
be `11`. This identifies lost centre-boundary lineage. No third density-memory
expansion was attempted.

## The periodic-background route

All 126 periodic free-boundary words of periods one through six close below
the predeclared 100,000-state cap. None has cycle density below `1/8`; the
minimum is `1/7`. This finite scan neither proves a uniform density bound nor
controls unbounded transients. Each fixed finite-period boundary does have
positive eventual density, by the zero-cycle/predecessor argument in
`docs/theory/periodic-boundary-density.md`.

The minimum has an exact physical seven-cell, four-time-step witness:

```text
1000000 -> 1100001 -> 0010011 -> 1111110 -> 1000000.
```

Column one alternates. Its initial left ones occur at zero-based bit indices
`0,7,14,...`. Any cone-charge inequality `alpha W_N-N >= -C` covering all
these infinite cones therefore needs `alpha >= 7`.

The finite truncations

```text
n_r = sum_(j=0..r-1) 2^(7j),  r >= 1,
```

admit a uniform interface argument. Odd `r` fail exactly at macrotime
`(7r-1)/2`. For even `r=2m`, an 18-bit moving-front system sampled every 14
half-steps has an exact 512-sample return cycle. Every arrival state fails
within eight macrosteps after switching to the prescribed centre boundary.
The exact delay table gives

```text
q(n_r) = 7r/2 + D(m mod 512),  0 <= D <= 8,
q(n_r) <= 4r  for every r >= 1.
```

The last bound uses the table for the seven small even cases, then
`7r/2+8 <= 4r` for `r>=16`. The infinite-family implication comes from the
closed recurrence returning to its initial state, not from a scan in `r`.
See `docs/theory/period7-ether-interface.md` for the coordinate and boundary
argument. This family is restricted: arbitrary finite rows remain unresolved.
The result concerns alternation starting from `n_r` at a centre-zero phase;
it does not control a later onset whose left half has left this family.

A targeted composition test refuted carrying the same eight-step allowance
through even one added bit. For `n_1181 OR 2^8254`, arrival is macrotime 4127,
and the low pairs at delays zero through eight are
`1,2,2,1,1,2,1,2,1`. Direct integer evolution confirms survival through 4135.
The horizon was not extended; a durable regression checks this exact witness.
Thus the next interface argument needs a compositional charge, not the same
constant allowance for every modified wake.

## Verification and scope

The independent density checker imports neither synthesizer. It reconstructs
the labeled inequalities from the rule truth table, verifies integer
cancellation, checks the genuine-cut obstruction, and rejects tampered
transitions, phases, coefficients, and scope. The periodic-boundary artifact
has a full deterministic replay command. Its Rule 15/90 controls concern the
finite-cone constructor; they do not validate the periodic orbit scanner.

The front-interface certificate has a separate truth-table verifier and a
hostile review. The verifier evolves ether and defect triples separately and
uses a shrinking truth-table cone for the post-arrival latch checks.

All 56 focused tests passed. Baseline and final serialized `verify_all` runs
each reported **8 passed, 3 expected skips, 0 failures**. The skipped inputs
are the two absent canonical bitstreams and the SAT proof toolchain; none is
used by these interface or local-constraint checks. A parallel development
run hit a worker-pool worktree-count assertion; the final serialized suite
passed. The first final run also caught a missing historical STATUS citation,
which was fixed before the clean rerun. No failed check is counted as a pass.

```bash
python tools/verify_coherent_density.py \
  runs/theory/coherent-density-obstructions.json \
  --temporal-pair-certificate runs/theory/temporal-pair-density.json
python tools/verify_period7_ether_interface.py \
  runs/theory/period7-ether-interface.json
python experiments/periodic_boundary_density.py \
  --verify runs/theory/periodic-boundary-density.json
python -m unittest tests.test_coherent_predecessor \
  tests.test_coherent_density_certificate tests.test_temporal_pair_density \
  tests.test_periodic_boundary_density tests.test_period7_ether_interface \
  tests.test_verify_coherent_density tests.test_verify_period7_ether_interface
python tools/verify_all.py
```

## Remaining obligation

The best next action is a **charge-preserving interface composition rule**:
account for how an added or removed initial one changes the wake and its delay
budget. It must survive the explicit one-bit counterexample and retain the
centre-boundary lineage lost by the density relaxations. The restricted
interface proof supplies an exact propagation mechanism; its fixed periodic
initial wake is the assumption that must be removed before arbitrary finite
configurations are covered. Neither another raw survival scan nor another
unmotivated memory increment addresses that assumption.

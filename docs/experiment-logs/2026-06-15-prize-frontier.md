# 2026-06-15 Prize Frontier

## Hypothesis

The center-column map `n -> center_bit(n)` might admit a compact digit automaton
or a compositional light-cone summary. A useful shortcut should survive finite
prefix checks with small DFAO state counts or show controlled summary growth with
collision-free composition at small cone depths.

## Prediction

- Positive control: Thue-Morse in base 2, MSD digit order, should recover a
  2-state DFAO on a 128-bit prefix.
- Rule 30 certificate track: if no small DFAO exists, generated SAT frontiers
  should be ready for UNSAT certificates over states 5-8, prefixes 128/256,
  bases 2/4, and MSD/LSD digit orders.
- Discovery track: if the first exact cone-summary family is promising, distinct
  summary counts should stay small and child summaries should compose without
  collisions as depth grows.

## Method

Implemented `prize_lab.py` commands:

- `dfao-frontier`: resumable exhaustive DFAO search artifacts with manifest
  entries.
- `dfao-sat-frontier`: DIMACS CNF plus JSON sidecars, optional solver command,
  and DIMACS count verification metadata.
- `cone-summary`: exact finite maps from boundary bits to final center-output
  windows, with composition-collision examples.
- `verify-artifacts`: manifest replay for sequence hashes, positive controls,
  DFAO candidates, and CNF header/body counts.

Commands run:

```bash
python prize_lab.py dfao-frontier --sequences thue-morse --directions msd --max-states 2 --run-id dfao-frontier-2026-06-15-positive --force
python prize_lab.py dfao-sat-frontier --run-id dfao-sat-frontier-2026-06-15-cnf
python prize_lab.py cone-summary --depths 4,5,6,7,8 --summary-widths 1,2,3,4 --out data/prize/2026-06-15-frontier/cone_summary_depths4-8_widths1-4.json --run-id cone-summary-2026-06-15
python prize_lab.py verify-artifacts data/prize/2026-06-15-frontier/manifest.json
```

## Controls

Positive control passed: `data/prize/2026-06-15-frontier/dfao_thue_morse_b2_msd_n128_s2.json`
recovers the expected 2-state Thue-Morse DFAO.

Null/baseline:

- SAT frontier sidecars use exact Rule 30 center prefixes and SHA-256 hashes.
- Cone summary compares Rule 30 against chaotic rule 45 and seeded random ECA
  rule 148 (`random_rule_seed=30`).

## Result

Artifact manifest:
`data/prize/2026-06-15-frontier/manifest.json`

- 32 DFAO SAT CNFs plus JSON sidecars generated for Rule 30 center prefixes:
  states 5-8, prefixes 128/256, bases 2/4, MSD/LSD.
- No SAT solver was supplied, so these are CNF artifacts and observations, not
  UNSAT certificates yet.
- Cone-summary Rule 30 depth-8 distinct summary counts were:
  width 1: 4, width 2: 25, width 3: 212, width 4: 324.
- Every checked Rule 30 composition case reported collisions; for example at
  depth 8, width 4 there were 122880 composition collisions.
- `verify-artifacts` checked 34 manifest entries and returned `ok: true`.

## Interpretation

Certificate track: the run archives the next SAT frontier but does not yet prove
new DFAO lower bounds beyond solved/exhaustive artifacts. A solver result of
UNSAT for a sidecar would promote that sidecar to a finite certificate.

Discovery track: the first boundary-map summary family does not compose cleanly
for Rule 30 at depths 4-8 and widths 1-4. This rules out this exact finite
summary design as an immediate shortcut, but not broader light-cone summaries.

## Next Promotion Step

Run `dfao-sat-frontier` with a SAT solver via `--solver-cmd`, archive SAT/UNSAT
statuses in the sidecars, and only promote UNSAT cases to DFAO lower-bound
certificates. For discovery, test richer summaries only if they define a precise
composition law before scaling depth.

## Continuation: Progress Bars and Prefix Scaling

Added stderr-only progress bars to long-running `prize_lab.py` experiment
commands. Stdout remains JSON; progress is visible in an interactive shell and
can be disabled with `--no-progress`.

Commands run:

```bash
python prize_lab.py dfao-frontier --run-id dfao-frontier-2026-06-15-s5 --progress
python prize_lab.py dfao-frontier --sequences center,random,thue-morse --bits 32,64,96,128 --max-states 4 --directions msd,lsd --run-id dfao-frontier-2026-06-15-prefix-scaling-s4 --progress
python prize_lab.py verify-artifacts data/prize/2026-06-15-frontier/manifest.json
```

Result:

- Exhaustive 5-state DFAO frontier finished for 128-bit prefixes, base 2,
  MSD/LSD.
- Rule 30 center: no 1-5 state DFAO fit for either digit direction.
- Seeded random baseline (`seed=30`): no 1-5 state DFAO fit for either digit
  direction.
- Thue-Morse control: 2-state DFAO recovered in both MSD and LSD directions.
- Prefix scaling through 4 states at 32/64/96/128 bits found no fits for Rule 30
  or the seeded random baseline in either digit direction.
- Thue-Morse recovered a 2-state DFAO at every tested prefix and direction.
- No local `kissat`, `cadical`, `minisat`, or `glucose` executable was found.
- `verify-artifacts` checked 64 manifest entries and returned `ok: true`.

Interpretation:

The 5-state result promotes the 128-bit center-column DFAO obstruction from
states 1-4 to states 1-5 by direct exhaustive search. Random matching the Rule
30 obstruction is expected and keeps the result framed as a finite certificate,
not a special randomness claim.

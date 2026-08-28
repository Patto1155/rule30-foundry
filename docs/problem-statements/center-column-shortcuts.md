# Rule 30 Center-Column Shortcut Artifacts

## Goal

Build exact tooling for Wolfram Rule 30 prize problem 2: decide whether the
single-black-cell center bit at time `n` can be computed by a shortcut that is
meaningfully smaller than simulating the cellular automaton to time `n`.

This repo should prefer artifacts that another agent can mechanically check:

- candidate programs/transducers/automata with a verifier command
- finite UNSAT encodings for named shortcut classes
- lower bounds for a named model class over a fixed tested prefix
- proof-candidate notes with explicit assumptions and gaps

## Current Entry Point

Use `prize_lab.py` for exact center-column work:

```bash
python prize_lab.py center --steps 256
python prize_lab.py recurrence --train-bits 512 --holdout-bits 128
python prize_lab.py kernel --steps 2048 --base 2 --depth 6 --sample-len 32
python prize_lab.py dfao-search --sequence center --bits 128 --max-states 4
python prize_lab.py dfao-sat --bits 64 --states 4 --out data/prize/dfao_b2_s4_n64.cnf
```

The output is JSON except `dfao-sat`, which writes DIMACS and prints metadata.

## Model Classes

Initial shortcut classes are deliberately small and checkable:

- GF(2) linear recurrences over the center-column sequence.
- Finite `k`-kernel signatures, giving lower bounds for LSB-first automatic
  sequence shortcuts over the tested finite domain.
- Exhaustive small-DFAO searches with positive controls.
- DFAO finite-prefix SAT encodings. `UNSAT` is a finite obstruction for the
  chosen base, direction, state count, and prefix length; `SAT` gives a candidate
  automaton that must be decoded and checked.

## Non-Goals

Do not treat another aggregate randomness test as prize progress unless it emits
one of the artifacts above or narrows a named shortcut class.

# New Rule 30 Research Idea — Agent Brief

You are working in the repository `Patto1155/rule30-foundry`.

Your task is to develop genuinely new, high-upside research ideas related to
Wolfram's three Rule 30 prize problems. You are not being asked merely to choose
the next item from the existing work queue. You may speculate, combine fields,
research external literature, and pursue ideas the project owner has probably
not considered.

Spend your time reasoning and researching, not wandering through the repository
or rebuilding tools that already exist.

## Load this context immediately

Before generating ideas, open and read these exact repository-relative files in
this order. Do not search for alternative handovers or older versions.

1. [`../CLAUDE.md`](../CLAUDE.md)
2. [`problem-statements/README.md`](problem-statements/README.md)
3. [`problem-statements/center-column-shortcuts.md`](problem-statements/center-column-shortcuts.md)
4. [`STATUS.md`](STATUS.md)
5. [`handover/CURRENT.md`](handover/CURRENT.md)
6. [`theory/README.md`](theory/README.md)
7. [`CLAIM_LEDGER.md`](CLAIM_LEDGER.md)
8. [`handover/2026-09-01-high-value-directions.md`](handover/2026-09-01-high-value-directions.md)
9. [`experiment-logs/2026-09-09-seven-step-followup.md`](experiment-logs/2026-09-09-seven-step-followup.md)
10. [`theory/finite-prefix-counting-bound.md`](theory/finite-prefix-counting-bound.md)
11. [`theory/periodic-trace-constraints.md`](theory/periodic-trace-constraints.md)
12. [`idea-bank/theoretical-reframe-2026-03-28.md`](idea-bank/theoretical-reframe-2026-03-28.md)

These files are the novelty and validity boundary. Treat proved facts, closed
routes, negative experiments and retracted claims as constraints. Files under
`docs/handover/archive/` are superseded and must not determine current state.

Only after loading the research context, run:

```bash
python tools/verify_all.py
```

Record every `SKIP`; it is not a `PASS`.

## Existing code and artifacts — reuse these

Do not rebuild basic Rule 30 simulation, generic CA exploration, model-class
searches or verification infrastructure. Inspect the named implementation only
when an idea actually needs it.

| Need | Existing path |
|---|---|
| Trusted packed Rule 30 operations | `experiments/rule30_open_utils.py` |
| Fast GPU simulation | `gpu/rule30_fast.py`, `gpu/rule30_sim.py` |
| Tape/light-cone sizing | `gpu/tape_geometry.py` |
| Arbitrary elementary CA controls | `experiments/eca_sim.py` |
| General CA simulation and coarse-graining | `ca_lab.py` |
| Centre-prefix, recurrence, kernel and DFAO utilities | `prize_lab.py` |
| Model-class admissibility/counting gate | `experiments/counting_bound.py` |
| Exact finite-prefix period search | `experiments/period_search_exact.py` |
| Exact DFAO searches | `experiments/dfao_min_states.py` |
| DFAO proof generation/verification | `experiments/dfao_drat_proofs.py`, `tools/verify_dfao_n56.py` |
| Grammar-compression probe | `experiments/grammar_min_size.py` |
| GF(2) annihilator search | `experiments/algebraic_annihilator.py` |
| Settled wedge and diagonal recurrence | `experiments/wedge_profile.py`, `experiments/diagonal_recursion.py` |
| Period doubling and pattern-map work | `experiments/period_doubling.py`, `experiments/pattern_map_walk.py` |
| Periodic traces and strip graphs | `experiments/periodic_trace_constraints.py`, `experiments/periodic_strip_graph.py` |
| Period-word and boundary searches | `experiments/period_word_sieve.py`, `experiments/periodic_right_boundary.py` |
| Actual-seed compatibility test | `experiments/seed_strip_compatibility.py` |
| Transient branch-selection probe | `experiments/transient_branch_selector.py` |
| Published shortcut audit | `experiments/nersissian_audit.py` |
| Existing result artifacts | `data/`, `data/prize/`, `runs/` |
| Repository verification | `tools/verify_all.py` |

Before writing a new tool, decide whether one of these can answer the question
with a new configuration, a small adapter or a disposable probe. Do not route a
deterministic existing script through an LLM.

The canonical `data/center_col_10M.bin` and `data/center_col_46M.bin` files may
be absent in a fresh checkout, although their hashes remain anchored. The
independently generated `data/golden/center_col_golden_10M.bin` is tracked and
available. Do not make recovering the 46M stream a prerequisite for a first
test unless the idea genuinely requires that scale.

## Existing results and routes not to rediscover

- No period `p <= 5,000,000` occurs in the first 10M centre bits, decided
  exactly. This does not decide eventual periodicity.
- Base-2 minimal DFAO complexity is certified through MSD `s*(56)=13`.
- Consecutive-window GF(2) searches found no annihilator of degree at most 3
  through width 64 or degree at most 4 through width 32.
- Existing LFSR, Markov, grammar-compression and neural-prediction work has not
  exposed a shortcut. The neural models also fail powered long-lag XOR controls,
  so generic additional neural prediction is low value.
- The Nersissian audit currently establishes only cheap warm queries after an
  `n`-dependent object exists. Its compressed masked/dyadic representation has
  not been faithfully reconstructed.
- Left diagonals obey a proved recurrence and are eventually periodic, but
  their periods are unbounded. The period-16 conjecture was refuted.
- The settled left wedge is compressible but never reaches the centre column.
- The diagonal pattern map is partial at zero words. Local continuations and
  locally valid strip cycles do not identify the actual single-seed branch.
- A small actual transient selects the correct known branch, but no bounded
  recurrence for its selector bit is known. Direct access to the first
  period-32 ambiguity near `d=1,420,878,969` is computationally infeasible.
- Tested nonconstant periodic centre words admit nonperiodic-neighbour strip
  loops. Merely enumerating more short words is not supported by the result.
- Entropy, bias, autocorrelation, block frequency, ordinary compression,
  transfer-style statistics, visual randomness and related diagnostics have
  already been explored extensively.
- Coarse-graining searches have not produced prize-facing structure.
- All three prizes concern the deterministic single-black-cell seed. Ensemble
  or random-initial-condition results are not prize progress by themselves.

Do not present one of these findings as a new idea. Do not simply scale it up
unless the larger run tests a named structural hypothesis.

## Research freedom

Look outside the repository's existing vocabulary. Potential connections
include symbolic dynamics, sofic shifts, automata and regular sequences,
communication complexity, branching programs, circuit lower bounds, semigroup
representations, transfer matrices, tensor networks, transducers with auxiliary
state, SAT interpolation, proof complexity, additive combinatorics, Boolean
semirings, renormalisation, substitutions, spacetime tilings, resource-bounded
Kolmogorov complexity, cryptanalytic distinguishers and automatic invariant
discovery. This list is suggestive, not restrictive.

An idea may recombine known results. For example, using the proved diagonal
recurrence to construct a new global single-seed invariant could be new even
though the recurrence itself is not.

The current `STATUS.md` queue records planned work; it does not restrict this
search. Prefer a better new direction if you find one.

## Method

1. Load the specified context.
2. Generate a broad private set of possibilities.
3. Reject anything already attempted, theoretically automatic, single-seed
   irrelevant, or vacuous under a counting argument.
4. Research the strongest unfamiliar connections using primary sources where
   possible.
5. Run cheap existing tools or write a small disposable probe to invalidate
   weak ideas quickly.
6. Do not begin a substantial implementation during ideation.
7. Select the strongest 3–5 genuinely new candidates.

## Required output

Rank 3–5 ideas by expected research value per day of work. For each provide:

1. **Core insight** — the proposed mechanism.
2. **Prize connection** — the exact prize question addressed.
3. **Novelty audit** — how it differs from documented work.
4. **Why it might work** — a concrete argument rather than an analogy.
5. **Fast falsification test** — the smallest experiment or proof check that
   could kill it.
6. **Existing components to reuse** — exact paths from this repository.
7. **Required new work** — only what must actually be added.
8. **Controls and failure modes** — including counting bounds, trivial local-rule
   rediscovery and actual single-seed compatibility.
9. **Success artifact** — candidate algorithm, invariant, certificate,
   obstruction or proof-shaped result.
10. **Estimated cost** — reasoning, implementation and compute.

Finish by recommending one idea for immediate investigation and specify a
tightly bounded first experiment. A valuable result need not immediately solve
a prize, but it must expose previously undocumented structure, eliminate a
genuinely plausible shortcut class, produce a reusable mathematical object, or
open a credible route toward a certificate or proof.

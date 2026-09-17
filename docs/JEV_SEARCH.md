# Jev search: choose the next finite question

One loop: **reviewed cards → choose → solve → verify → prune → choose again**.

The first lane is the base-2 DFAO frontier identified in `STATUS.md`. A card
asks whether an automaton of at most `s` states reproduces the first `n` bits.
Jev chooses which unresolved card gets the next solve budget. It sees checked
outcomes from previous steps. It cannot supply commands, edit cards, declare a
proof valid, or promote a claim. The lead agent owns the hypotheses.

This is deliberately one research lane, primarily relevant to Prize 3. It
does not manufacture a Prize 2 experiment merely to cover all three prizes.
Finite automaticity results may inform proof ideas; they do not settle any
asymptotic prize. Official numbering: 1 nonperiodicity, 2 equidistribution,
3 computational effort for the nth center bit.

## Start

From the repository root, Python + the existing NumPy dependency:

```bash
python tools/verify_all.py
bash tools/build_sat_toolchain.sh
python experiments/dfao_drat_proofs.py --self-test
python tools/jev_search.py queue/jev/smoke.json --policy fixed --out runs/jev/smoke
```

The small plan is **calibration replay**, not new research. It checks SAT,
UNSAT, controls and monotone pruning. Its frontier-progress count stays zero.

The frontier plan asks about n=64 at 13/14 states on the center column and a
matched random sequence, plus 15 states for the random sequence. Three
controls run first: a Thue-Morse witness, a known UNSAT instance, and the
existing center-column 15-state upper bound. The known upper bound earns no
frontier credit. Missing tools or a failed control stop the run.

```bash
python tools/jev_search.py queue/jev/frontier.json --policy fixed --out runs/jev/fixed
python tools/jev_search.py queue/jev/frontier.json --policy random --seed 30 --out runs/jev/random
python tools/jev_search.py queue/jev/frontier.json --policy jev --out runs/jev/jev
```

The last command requires `TYPESAFE_API_KEY` in the environment. It makes real
requests to TypeSafe; there is no mock fallback. All three use the same plan,
solve/check budgets and stopping rules. Each output directory must be new.
Run `verify_all.py` after the experiment, as required by `AGENTS.md`.
Per-card preflight records external integrity stages as SKIP: these are the
session's separate `verify_all` responsibility, not silently claimed passes.

## What actually produces evidence

- Center bits are regenerated and checked against the independent naive
  generator before encoding; canonical packed files are not needed.
- `prize_lab.dfao_sat_cnf` supplies the existing encoding.
- SAT: decode CaDiCaL's model, then evaluate the DFAO with Foundry's independent
  evaluator against every requested bit.
- UNSAT: retain the CNF and DRAT proof and require `drat-trim` acceptance.
- Timeouts are UNKNOWN. A rejected proof stops the session. No unverified
  result can prune a later question.
- Checked SAT at `(n,s)` implies SAT at shorter prefixes and larger state
  caps. Checked UNSAT implies UNSAT at longer prefixes and smaller caps.
  Pruning never crosses sequence, random seed, base or digit direction.
- A small-class negative is admitted only as a bounded exact exclusion via
  the existing counting gate. Its truth is distinct from discriminatory power.

Replay a specific saved result, recomputing the input and CNF:

```bash
python tools/jev_search.py --verify-result runs/jev/smoke/control-tm64-s2
python tools/jev_search.py --verify-result runs/jev/smoke/control-center16-s4
```

The replay rejects modified CNFs, invalid witnesses, changed proof hashes and
invalid proofs. Solver/checker logs and all artifacts stay in the run folder.
It is a finite-instance checker, not a full `s*(n)` certificate assembler:
missing lower bounds remain missing, even if an upper bound is found.

## Jev's role and limits

One call presents all eligible cards and the checked results. Its Choice
answer must select an offered ID (or `return-to-lead`) and carry a valid
distribution. A selected probability below `--min-probability` stops for lead
review. The default 0.5 is an experimental routing threshold, not a measured
accuracy guarantee. No repeated voting or model-written proof claims.

Defaults: 120-second session, 10-second solver, 10-second checker, 12 attempted
instances, 10 Jev calls and 327,680 input tokens. Subprocess calls use the
remaining session budget. Python input/CNF construction is bounded by the
plan limits but is not forcibly interrupted; the deadline can overrun during
that work. Model time counts against the same session budget as solver time.
Token accounting reserves 32,768 input tokens before each call and keeps that
reserve on an API failure, so failed calls do not become free in the ledger.
The USD estimate uses the published $0.042/M input price, not an invoice.

Plans contain at most 200 unique instances, n<=256, s<=24, bases 2..4. They
require SAT/UNSAT calibration, and a matching random card for every center
frontier card. Only fixed typed solver operations execute. New scientific
families need a reviewed adapter and verifier; do not label unrelated shell
commands as experiments. All attempts count against the step budget, including
timeouts. Retry a hard case deliberately in a new plan with a larger budget.

## Does Jev help?

Compare identical plans with fixed/random/Jev policies. Measure:

1. Newly checked center frontier instances per total second, including Jev.
2. Bounds tightened and which exact cases remain unresolved.
3. Solver calls avoided by sound implications, shown separately from direct proofs.
4. Model tokens, errors, unknowns and elapsed time.

`summary.json` separates controls, random cases, direct center results and
inferred answers. The label `frontier` is supplied by the reviewed plan; it
does not establish novelty. Retire solved cards after checking STATUS/ledger.
Keep every run, including failures; repeat with several fixed random seeds
and report variable solver time. A tiny replay validates the wiring only.
The agent's existing ordering can be the fixed policy: save that order before
running. Do not select the best run after seeing its results.

There is no reason to spend thousands of calls on a handful of exact cases.
If all useful cards are resolved or Jev returns to the lead, formulate a new
testable hypothesis. The next substantial adapter could cover sparse-lag
annihilators (C2c); that requires its own scientific implementation and gates.

API contract checked 2026-09-17:
[TypeSafe quickstart](https://docs.typesafe.ai/introduction/quickstart),
[Choice/HTTP reference](https://docs.typesafe.ai/api),
[pricing and launch caveats](https://typesafe.ai/blog/introducing-system-one-models-and-jev).
